#include <mpi.h>
#include <stdio.h>
#include <pthread.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <assert.h>
#include <getopt.h>

enum {
    TEST_BARRIER,
    TEST_BCAST,
    TEST_ALLREDUCE,
    TEST_REDUCE,
    TEST_ALLGATHER,
    TEST_LAST,
};

const char *coll_names[TEST_LAST] = {
    "BARRIER", "BCAST", "ALLREDUCE", "REDUCE", "ALLGATHER" };

struct stat{
    int num_calls_small[TEST_LAST];
    int num_calls_large[TEST_LAST];
    double elapsed_small[TEST_LAST];
    double elapsed_large[TEST_LAST];
};

struct params {
    int n_threads;
    int n_splits;
    int tests_per_comm;
    int coll_include;
    int coll_exclude;
    int barrier_after_split;
} P = {4, 1000, 100, -1, -1, 0};

struct stat stats;

typedef enum {
    TYPE_INT,
    TYPE_DOUBLE
} mpi_datatype_t;

#define GET_TYPE(_test_data) (_test_data == TYPE_INT ? MPI_INT : MPI_DOUBLE)
struct test_data {
    mpi_datatype_t type;
    int count;
    int is_large;
    int coll;
    int split_color;
    int id;
    double elapsed;
};

struct thread_data {
    pthread_t tid;
    MPI_Comm comm;
    int running;
    struct test_data *test;
};

#define SET_VALUE(_buf, _type, _pos, _value) do{        \
        if (MPI_INT == _type) {                   \
            ((int*)_buf)[_pos] = (int)_value; \
        } else {                                        \
            ((double*)_buf)[_pos] = (double)_value;     \
        }                                               \
    } while(0)


void init_buffers(void *sbuf, void *rbuf, MPI_Datatype type,
                  int count, int coll, int root, int rank) {
    int i;
    size_t lb, extent;
    MPI_Type_get_extent(type, &lb, &extent);
    switch(coll) {
    case TEST_BCAST:
        if (rank == root) {
            for (i=0; i<count; i++) {
                SET_VALUE(sbuf, type, i, 0xdeadbeef);
            }
        } else {
            memset(sbuf,0,count*extent);
        }
        break;
    case TEST_ALLREDUCE:
    case TEST_REDUCE:
    case TEST_ALLGATHER:
        memset(rbuf,0,count*extent);
        for (i=0; i<count; i++) {
            SET_VALUE(sbuf, type, i, (rank+1));
        }
        break;
    default:
        break;
    }
}


#define CHECK_VALUE(_buf, _type, _pos, _value) do{                      \
        if (MPI_INT == _type) {                                   \
            if (((int*)_buf)[_pos] != (int)_value) {          \
                fprintf(stderr, "CORRECTNESS ERROR: id %d, TEST_TYPE %d, pos %d, value %d, expected %d, dtype %s, root %d, rank %d, count %d, comm_size %d, color %d\n", \
                        id, coll, _pos, ((int*)_buf)[_pos], (int)_value, "MPI_INT", root, rank, count, comm_size, color); \
                MPI_Abort(MPI_COMM_WORLD, -1);                             \
            }                                                           \
        } else {                                                        \
            if (fabs(((double*)_buf)[_pos] - (double)_value) > 1e-3) {  \
                fprintf(stderr, "CORRECTNESS ERROR: id %d, TEST_TYPE %d, pos %d, value %g, expected %g, dtype %s, root %d, rank %d, count %d, comm_size %d, color %d\n", \
                        id, coll, _pos, ((double*)_buf)[_pos], (double)_value, "MPI_DOUBLE", root, rank, count, comm_size, color); \
                MPI_Abort(MPI_COMM_WORLD, -1);                             \
            }                                                           \
        }                                                               \
    }while(0)

void check_buffers(void *sbuf, void *rbuf, MPI_Datatype type,
                   int count, int coll, int root, int rank, int comm_size, int id, int color) {
    int i;
    int rst = (1+comm_size)*comm_size/2;
    switch(coll) {
    case TEST_BCAST:
        if (rank != root) {
            for (i=0; i<count; i++) {
                CHECK_VALUE(sbuf, type, i, 0xdeadbeef);
            }
        }
        break;
    case TEST_ALLREDUCE:
        for (i=0; i<count; i++) {
            CHECK_VALUE(rbuf, type, i, rst);
        }
        break;
    case TEST_REDUCE:
        if (rank == root) {
            for (i=0; i<count; i++) {
                CHECK_VALUE(rbuf, type, i, rst);
            }
        }
        break;
    case TEST_ALLGATHER:
        for (i=0; i<count*comm_size; i++) {
            rst = i/count + 1;
            CHECK_VALUE(rbuf, type, i, rst);
        }
        break;
    default:
        break;
    }
}

void* start_thread(void *arg) {
    struct thread_data *thread = (struct thread_data*)arg;
    int i;
    void *sbuf = NULL, *rbuf = NULL;
    size_t lb, extent;
    MPI_Datatype type = GET_TYPE(thread->test);
    int coll = thread->test->coll;
    int count = thread->test->count;
    MPI_Comm comm = thread->comm;
    int comm_size, rank, root = 0;
    MPI_Comm_size(comm, &comm_size);
    MPI_Comm_rank(comm, &rank);
    assert(type == MPI_INT || type == MPI_DOUBLE);
    MPI_Type_get_extent(type, &lb, &extent);
    double t_start = MPI_Wtime();
    if (thread->test->is_large) {
        stats.num_calls_large[coll]++;
    } else {
        stats.num_calls_small[coll]++;
    }
    switch (coll) {
    case TEST_BCAST:
        sbuf = (void*)malloc(count*extent);
        break;
    case TEST_ALLREDUCE:
    case TEST_REDUCE:
        sbuf = (void*)malloc(count*extent);
        rbuf = (void*)malloc(count*extent);
        break;
    case TEST_ALLGATHER:
        sbuf = (void*)malloc(count*extent);
        rbuf = (void*)malloc(count*extent*comm_size);
        break;
    default:
        break;
    };
    for (i=0; i<P.tests_per_comm; i++) {
        init_buffers(sbuf,rbuf,type,count,coll,root,rank);
        switch (coll) {
        case TEST_BARRIER:
            MPI_Barrier(comm);
            break;
        case TEST_BCAST:
            MPI_Bcast(sbuf,count,type,root,comm);
            break;
        case TEST_ALLREDUCE:
            MPI_Allreduce(sbuf,rbuf,count,type,MPI_SUM,comm);
            break;
        case TEST_REDUCE:
            MPI_Reduce(sbuf,rbuf,count,type,MPI_SUM,root,comm);
            break;
        case TEST_ALLGATHER:
            MPI_Allgather(sbuf,count,type,rbuf,count,type,comm);
            break;
        default:
            break;
        };
        check_buffers(sbuf,rbuf,type,count,coll,root,rank, comm_size, thread->test->id, thread->test->split_color);
        root = (root + 1) % comm_size;
    }
    if (sbuf) free(sbuf);
    if (rbuf) free(rbuf);
    thread->test->elapsed = MPI_Wtime() - t_start;
    thread->running = 0;
    return NULL;
}

void spawn_new(struct thread_data *thread, struct test_data *test, int world_rank) {
    assert(thread->running == 0);
    thread->running = 1;
    if (thread->comm != MPI_COMM_NULL) {
        MPI_Comm_free(&thread->comm);
        assert(thread->comm == MPI_COMM_NULL);
        if (thread->test->is_large) {
            stats.elapsed_large[thread->test->coll] += thread->test->elapsed;
        } else {
            stats.elapsed_small[thread->test->coll] += thread->test->elapsed;
        }
    }

    MPI_Comm_split(MPI_COMM_WORLD, test->split_color, world_rank, &thread->comm);
    MPI_Barrier(MPI_COMM_WORLD);
    if (P.barrier_after_split) {
        MPI_Barrier(thread->comm);
    }
    assert(thread->comm != MPI_COMM_NULL);
    if (world_rank == 0) {
        printf("Splitting id %d\t\t\r", test->id);
        fflush(stdout);
    }
    thread->test = test;
    pthread_create(&thread->tid, NULL, start_thread, (void*)thread);
}

static void check_incl_excl(int *ci, int *ce, char *var_i, char *var_e) {
    int i;
    if (var_i && var_e) {
        fprintf(stderr, "ERROR: MTS_COLL_INCL and MTS_COLL_EXCL are set simultaneously. Choose only one.\n");
        exit (-1);
    }

    *ci = *ce = -1;

    if (var_i) {
        for (i=0; i<TEST_LAST; i++) {
            if (!strcmp(var_i, coll_names[i])) {
                *ci = i;
                break;
            }
        }
    }

    if (var_e) {
        for (i=0; i<TEST_LAST; i++) {
            if (!strcmp(var_e, coll_names[i])) {
                *ce = i;
                break;
            }
        }
    }
}

static void parse_args(int argc, char **argv) {
    int c;
    char coll_include[64], coll_exclude[64];
    char *ci = NULL, *ce = NULL;
    while (1)
    {
              static struct option long_options[] =
              {
                  {"n-threads",  required_argument, 0, 't'},
                  {"n-splits",   required_argument, 0, 's'},
                  {"tests-per-comm",    required_argument, 0, 'c'},
                  {"coll-include",    required_argument, 0, 'i'},
                  {"coll-exclude",    required_argument, 0, 'e'},
                  {"barrier-after-split",    required_argument, 0, 'b'},
                  {0, 0, 0, 0}
              };
              /* getopt_long stores the option index here. */
              int option_index = 0;

              c = getopt_long (argc, argv, "t:s:c:i:e:b:",
                               long_options, &option_index);

              /* Detect the end of the options. */
              if (c == -1)
                  break;

              switch (c)
              {
              case 't':
                  P.n_threads = atoi(optarg);
                  break;

              case 's':
                  P.n_splits = atoi(optarg);
                  break;

              case 'c':
                  P.tests_per_comm = atoi(optarg);
                  break;

              case 'b':
                  P.barrier_after_split = atoi(optarg);
                  break;

              case 'i':
                  strcpy(coll_include, optarg);
                  ci = coll_include;
                  break;

              case 'e':
                  strcpy(coll_exclude, optarg);
                  ce = coll_exclude;
                  break;

              case '?':
                  /* getopt_long already printed an error message. */
                  break;

              default:
                  abort ();
              }
    }
    check_incl_excl(&P.coll_include, &P.coll_exclude, ci, ce);
}
int main(int argc, char *argv[])
{
    int provided;
    int comms_tested = 0;
    int rand_seed;
    int rank, i, size;
    void *retval;

    struct thread_data *threads = (struct thread_data*)calloc(sizeof(*threads), P.n_threads);
    struct test_data *tests = (struct test_data*)calloc(sizeof(*tests), P.n_splits);

    for (i=0; i < P.n_threads; i++) {
        threads[i].comm = MPI_COMM_NULL;
    }

    parse_args(argc, argv);
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    double elapsed = MPI_Wtime();
    if (provided != MPI_THREAD_MULTIPLE) {
        fprintf(stderr, "ERROR: MPI does not support Multi-threaded mode\n");
        return -1;
    }

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (0 == rank) {
        memset(&stats,0,sizeof(stats));
        for (i=0; i<TEST_LAST; i++) {
            stats.num_calls_large[i] = 0;
            stats.num_calls_small[i] = 0;
            stats.elapsed_small[i] = 0;
            stats.elapsed_large[i] = 0;
        }
        rand_seed = argc > 1 ? atoi(argv[1]) : (int)(time(NULL));
        printf("************************************************************************\n");
        printf("* RAND SEED %d, N_THREADS %d, N_SPLITS %d, TESTS_PER_COMM %d \n",
               rand_seed, P.n_threads, P.n_splits, P.tests_per_comm);
        printf("************************************************************************\n");
        srand(rand_seed);

        for (i=0; i<P.n_splits; i++) {
            int rand_value = rand();
            double rand_value_f = (double)rand_value/(double)RAND_MAX;
            if (P.coll_include >= 0) {
                tests[i].coll = P.coll_include;
            } else if (P.coll_exclude >= 0) {
                int coll = rand_value % (TEST_LAST - 1);
                if (coll >= P.coll_exclude) {
                    coll++;
                }
                tests[i].coll = coll;
            } else {
                tests[i].coll = rand_value % TEST_LAST;
            }
            int small_count_low = 1;
            int small_count_high = 128;
            int large_count_low = 8192;
            int large_count_high = large_count_low*2;
            tests[i].type = rand_value_f > 0.5 ? TYPE_INT : TYPE_DOUBLE;
            tests[i].id = i;
            if (tests[i].coll == TEST_ALLGATHER) {
                large_count_low = (int)((double)8192/size);
                large_count_high = large_count_low*2;
            }

            tests[i].is_large = 0;
            tests[i].count = (int)(small_count_low
                                   + (small_count_high-small_count_low)*rand_value_f);

            if (rand_value_f > 0.5) {
                tests[i].is_large = 1;
                tests[i].count = (int)(large_count_low
                                       + (large_count_high-large_count_low)*rand_value_f);
            }
        }
        int *split_colors = (int*)malloc(sizeof(int)*P.n_splits*size);
        for (i=0; i<P.n_splits*size; i++) {
            double rand_value_f = (double)rand()/(double)RAND_MAX;
            split_colors[i] = rand_value_f > 0.5 ? 1 : 2;
        }
        MPI_Bcast(tests, sizeof(*tests)*P.n_splits, MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Scatter(split_colors, P.n_splits, MPI_INT,
                    MPI_IN_PLACE, P.n_splits, MPI_INT, 0, MPI_COMM_WORLD);

        for (i=0; i<P.n_splits; i++) {
            tests[i].split_color = split_colors[i];
        }
        
        free(split_colors);

    } else {
        int *split_colors = (int*)malloc(sizeof(int)*P.n_splits);
        MPI_Bcast(tests, sizeof(*tests)*P.n_splits, MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Scatter(MPI_IN_PLACE, P.n_splits, MPI_INT,
                    split_colors, P.n_splits, MPI_INT, 0, MPI_COMM_WORLD);

        for (i=0; i<P.n_splits; i++) {
            tests[i].split_color = split_colors[i];
        }
        
        free(split_colors);
    }

    while (comms_tested < P.n_splits) {
        for (i=0; i<P.n_threads && comms_tested < P.n_splits; i++) {
            while (threads[i].running) {
                usleep(50);
            }
            spawn_new(&threads[i], &tests[comms_tested], rank);
            comms_tested++;
        }
    }

    for (i=0; i<P.n_threads; i++) {
        pthread_join(threads[i].tid, &retval);
        if (threads[i].comm != MPI_COMM_NULL) {
            MPI_Comm_free(&threads[i].comm);
            threads[i].comm = MPI_COMM_NULL;
        }

    }
    MPI_Barrier(MPI_COMM_WORLD);

    free(tests);
    free(threads);
    elapsed = MPI_Wtime() - elapsed;
    MPI_Finalize();
    if (0 == rank) {
        fprintf(stdout,"\nALL DONE: elapsed %.1f sec\n===========================================================\n", elapsed);
        for (i=0; i<TEST_LAST; i++) {
            printf("%-20s [small]: %-10d took %-6.1f sec\n", coll_names[i],
                   stats.num_calls_small[i], stats.elapsed_small[i]);
            printf("%-20s [large]: %-10d took %-6.1f sec\n", coll_names[i],
                   stats.num_calls_large[i], stats.elapsed_large[i]);
        }
    }
    return 0;
}
