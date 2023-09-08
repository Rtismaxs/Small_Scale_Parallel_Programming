# MAKEFILE FOR OPENMP
# Matrix-vector product loop unrolling by a factor of 4

AMVU4=a_matrix_vector_unroll_4.o wtime.o mmio.o

LFLAGS=-fopenmp
CFLAGS=-O4 -fopenmp
     
all: mmio.o wtime.o a_matrix_vector_unroll_4
 
a_matrix_vector_unroll_4: $(AMVU4)
	$(CC) $(LFLAGS) -o$@ $(AMVU4)

clean:
	/bin/rm -fr *.o

.c.o:
	$(CC) $(CFLAGS) -c -o$@ $< 
