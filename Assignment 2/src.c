#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Macro for flat 1D array indexing treating memory as [Z][Y][X][F]
// The grid is stored as a 1D array, all the dimensions flattened. So given x, y, z, f coordinates, the array index is calculated as follows:
// Innermost dimension is field f, hence it is added as it is as the field offset.
// There are F number of fields for each (x, y, z) point, hence F is multiplied with total number of points in each plane to get the field index.
// x is the offset in the row, similarly, there will be y rows of x-dim points(i.e. X_dim), hence y*X_dim,
// and for z dimension, there will be z planes of (Y_dim*X_dim) points, hence z*(Y_dim*X_dim).
// Hence, for any (x, y, z) point, the total number of points to be traversed are z*(Y_dim*X_dim) + y*(X_dim) + x,
// where X_dim and Y_dim are the dimensions of the local grid including ghost cells (nx+2*w and ny+2*w respectively).
#define IDX(x, y, z, f) ((((z) * Y_dim + (y)) * X_dim + (x)) * F + (f))

// Beginning of main function
int main(int argc, char **argv)
{
    // Declaring the rank of the process and total number of processes
    int myrank, num_procs;
    // Initializing the MPI environment using MPI_Init
    MPI_Init(&argc, &argv);
    // Getting the rank of the process using MPI_Comm_rank
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    // Getting the total number of processes using MPI_Comm_size
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // --- 1. INPUT PARSING ---
    // Error handling for number of input arguments
    if (argc != 13)
    {
        // Only rank 0 will print the error message to avoid clutter
        if (myrank == 0)
            // Print the usage message with expected arguments
            printf("Usage: %s d ppn px py pz nx ny nz T seed F isovalue\n", argv[0]);
        // Finalize the MPI environment before exiting
        MPI_Finalize();
        // Return due to error
        return 1;
    }

    // Parsing input arguments from command line
    // Distance for stencil computation
    int d = atoi(argv[1]);
    // Number of processes allocated per node
    int ppn = atoi(argv[2]);
    // Dimensions of the process grid
    int px = atoi(argv[3]);
    int py = atoi(argv[4]);
    int pz = atoi(argv[5]);
    // Local grid dimensions of a single subdomain (excluding ghost cells)
    int nx = atoi(argv[6]);
    int ny = atoi(argv[7]);
    int nz = atoi(argv[8]);
    // Number of time steps for the simulation
    int T = atoi(argv[9]);
    // Seed for random data generation
    int seed = atoi(argv[10]);
    // Number of fields in the simulation
    int F = atoi(argv[11]);
    // Isovalue for counting isovalue crossings
    double isoval = atof(argv[12]);

    // Error handling to ensure total number of processes matches the process grid dimensions
    if (num_procs != px * py * pz)
    {
        // Only rank 0 will print the error message to avoid clutter
        if (myrank == 0)
            // Print the error message indicating the mismatch in total processes and process grid dimensions
            printf("ERROR: Total processes (%d) != px*py*pz (%d)\n", num_procs, px * py * pz);
        // Finalize the MPI environment before exiting
        MPI_Finalize();
        // Return due to error
        return 1;
    }

    // Generalized stencil reach (width 'w')
    // w = Depth of ghost cells (assuming d is always of the form 6k+1, where k>0)
    // d is the total neighbours to be considered in the stencil, and there are 6 directions (east, west, north, south, front, back) times k (layer width in each direction), + 1 for the point itself
    // so width w = k = (d-1)/6
    int w = (d - 1) / 6;

    // Ensure that w is at least 1
    if (w < 1)
        w = 1;

    // --- 2. TOPOLOGY & DOMAIN ---
    // Calculate the 3D coordinates of the current process in the process grid
    // The process grid is organized in a 3D manner, where the rank mapping of the process to its (x, y, z) coordinates in the grid is as follows.
    // For z-coordinate, each plane of processes in the z-dimension contains (px*py) processes, hence z = myrank/(px*py).
    int my_z = myrank / (px * py);
    // For x and y coordinates, we first 'eliminate' the current plane offset (rem) in the z-dimension using myrank % (px * py), and then map it to y and x coordinates.
    // As simply myrank/px (for Y) will include the number of planes travelled in z-dimension as well, we take modulo py to reset y after every py rows.
    // For x coordinate, since it resets after every px processes, we can directly take modulo px.
    int rem = myrank % (px * py);
    // For y coordinate, the plane has px processes in each row, so y coordinate is rem/px
    int my_y = rem / px;
    // For x coordinate, since it resets after every px processes, we can directly take modulo px.
    int my_x = rem % px;

    // Calculating the ranks of the 6 neighbors (east, west, north, south, front, back) using the coordinates and process grid dimensions
    // For west neighbor, check if process is not in the first column (my_x > 0), then west neighbor is simply myrank - 1, otherwise null
    int rank_west = (my_x > 0) ? myrank - 1 : MPI_PROC_NULL;
    // For east neighbor, check if process is not in the last column (my_x < px - 1), then east neighbor is simply myrank + 1, otherwise null
    int rank_east = (my_x < px - 1) ? myrank + 1 : MPI_PROC_NULL;
    // For south neighbor, check if process is not in the first row (my_y > 0), then south neighbor is myrank - px (one row above), otherwise null
    int rank_south = (my_y > 0) ? myrank - px : MPI_PROC_NULL;
    // For north neighbor, check if process is not in the last row (my_y < py - 1), then north neighbor is myrank + px (one row below), otherwise null
    int rank_north = (my_y < py - 1) ? myrank + px : MPI_PROC_NULL;
    // For back neighbor, check if process is not in the first plane (my_z > 0), then back neighbor is myrank - (px * py) (one plane behind), otherwise null
    int rank_back = (my_z > 0) ? myrank - (px * py) : MPI_PROC_NULL;
    // For front neighbor, check if process is not in the last plane (my_z < pz - 1), then front neighbor is myrank + (px * py) (one plane in front), otherwise null
    int rank_front = (my_z < pz - 1) ? myrank + (px * py) : MPI_PROC_NULL;

    // Calculate the local grid dimensions including ghost cells
    // w is the width of the ghost cell layer, and nx, ny and nz are the dimensions of the local grid (excluding ghost cells)
    // The grid is sandwiched between ghost cell layers, so the starting with 0 index till w is a ghost layer, so the grid originally starts from w, goes till t+w.
    // The final ghost cell layer are from t+w to t+2*w, where t is any of nx, ny, nz. The grid assumes indexing as follows.
    // 0-W are starting ghost cells in each dimension, t+w to t+2*w are ending ghost cells for t in {nx, ny, nz}.
    // the ghost cell layers are on the boundary of each face, hence for a particular dimension, there are 2 valid faces (x->East,West; y->North,South; z->Front,Back)
    // Converting 3D coordinates to the 1D index w.r.t ghost cell offset. x as it is, for y, need to traverse nx(original) + 2*w(ghost cells) for each row.
    // Similarly for z, need to traverse (nx+2*w)*(ny+2*w) cells for each plane.
    int X_dim = nx + 2 * w;
    int Y_dim = ny + 2 * w;
    int Z_dim = nz + 2 * w;

    // Total number of cells in the local grid including ghost cells for all fields
    long long total_cells = (long long)X_dim * Y_dim * Z_dim * F;

    // --- 3. MEMORY ALLOCATION ---
    // Allocating memory for the current time step data
    double *data_current = calloc(total_cells, sizeof(double));
    // Allocating memory for the next time step data (used for stencil computation to avoid overwriting current data)
    double *data_next = calloc(total_cells, sizeof(double));

    // --- 4. INITIALIZATION (Matching exact loop constraints) ---
    // Initialize the random number generator using the provided seed
    srand(seed);
    // Initialize the grid data for each field with random values using the provided formula
    for (int f = 0; f < F; f++)
    {
        // For each field, the data_current corresponds to a 3D grid of dimensions (nx, ny, nz) is initialized
        for (int iter = 0; iter < nx * ny * nz; iter++)
        {
            // The value to be initialized to a grid point is calculated using the formula provided in the assignment
            double val = (double)rand() * (myrank + 1) / (110426.0 + f + iter);
            // Map the 1D index 'iter' to 3D interior coordinates (x_int, y_int, z_int)
            // For z coordinate, each plane has nx*ny points, hence z_int = iter/(nx*ny)
            int z_int = iter / (nx * ny);
            // For x and y coordinate, the offset in a single plane is rem_iter = iter % (nx*ny),
            // As simply iter/nx (for Y) will include the number of planes travelled in z-dimension as well, we take modulo nx*ny to reset y after every py rows.
            int rem_iter = iter % (nx * ny);
            // Since each row has nx points, hence y_int = (rem_iter)/nx
            int y_int = rem_iter / nx;
            // For x coordinate, since it resets after every nx elements, we can directly take modulo nx.
            int x_int = rem_iter % nx;
            // Store the initialized value in the data_current array at the correct index corresponding to (x_int, y_int, z_int, f) coordinates, offset by w to account for ghost cells
            data_current[IDX(x_int + w, y_int + w, z_int + w, f)] = val;
        }
    }

    // --- 5. MPI DATATYPE DEFINITIONS (MPI_Type_indexed) ---
    // Creating MPI Indexed datatypes for ghost layer communication(halo exchange) for each face using a single send/recv call per face
    MPI_Datatype type_x_send_west, type_x_recv_west, type_x_send_east, type_x_recv_east;
    // We need to send ny*nz*w*F cells for each face. So for the displacement and block length arrays, we will have ny*nz count, while block lengths will be w*F
    // Thereby making total ny*nz*w*F cells, because in the original data grid, F is the innermost dimension and w is the width of ghost layer,
    // so we can get w*F elements in a contiguous manner.
    int count_x = ny * nz;
    // Allocating block length and displacement arrays for the 4 faces in x dimension (west and east send and receive)
    int *blen_x = malloc(count_x * sizeof(int));
    int *disp_x_sw = malloc(count_x * sizeof(int));
    int *disp_x_rw = malloc(count_x * sizeof(int));
    int *disp_x_se = malloc(count_x * sizeof(int));
    int *disp_x_re = malloc(count_x * sizeof(int));

    // Iterator for the block length and displacement arrays
    int c = 0;
    // Looping over the y and z dimensions to fill the block length and displacement arrays for x dimension communication
    for (int k = w; k < nz + w; k++)
    {
        for (int j = w; j < ny + w; j++)
        {
            // each block length is w*F, as we can send/recv w layers of ghost cells for all F fields in a contiguous manner in the original data grid.
            blen_x[c] = w * F;
            // For west face send, the starting point is at x=w, because 0 to w-1 are ghost cells.
            disp_x_sw[c] = IDX(w, j, k, 0);
            // For west face recv, the starting point is at x=0, to copy into the ghost cells on west face
            disp_x_rw[c] = IDX(0, j, k, 0);
            // For east face send, the starting point is at x=nx, because nx to nx+w-1 are the ghost cells on east face.
            disp_x_se[c] = IDX(nx, j, k, 0);
            // For east face recv, the starting point is at x=nx+w, to copy into the ghost cells on east face
            disp_x_re[c] = IDX(nx + w, j, k, 0);
            // Incrementing the iterator for block length and displacement arrays
            c++;
        }
    }
    // Creating MPI indexed datatypes for x dimension communication using the block length and displacement arrays and committing them
    MPI_Type_indexed(count_x, blen_x, disp_x_sw, MPI_DOUBLE, &type_x_send_west);
    MPI_Type_commit(&type_x_send_west);
    MPI_Type_indexed(count_x, blen_x, disp_x_rw, MPI_DOUBLE, &type_x_recv_west);
    MPI_Type_commit(&type_x_recv_west);
    MPI_Type_indexed(count_x, blen_x, disp_x_se, MPI_DOUBLE, &type_x_send_east);
    MPI_Type_commit(&type_x_send_east);
    MPI_Type_indexed(count_x, blen_x, disp_x_re, MPI_DOUBLE, &type_x_recv_east);
    MPI_Type_commit(&type_x_recv_east);

    // Similarly, creating MPI indexed datatypes for y dimension communication.
    MPI_Datatype type_y_send_south, type_y_recv_south, type_y_send_north, type_y_recv_north;
    // For y dimension, we need nx*nz*w*F cells for each face, so the count for block length and displacement arrays will be nz*w,
    // and each block length will be nx*F as we can send/recv an entire row of x dimension for all F fields in a contiguous manner in the original data grid.
    int count_y = nz * w;
    // Allocating block length and displacement arrays for the 4 faces in y dimension (south and north send and receive)
    int *blen_y = malloc(count_y * sizeof(int));
    int *disp_y_ss = malloc(count_y * sizeof(int));
    int *disp_y_rs = malloc(count_y * sizeof(int));
    int *disp_y_sn = malloc(count_y * sizeof(int));
    int *disp_y_rn = malloc(count_y * sizeof(int));

    // Iterator for the block length and displacement arrays
    c = 0;
    // Looping over the z and y dimensions to fill the block length and displacement arrays for y dimension communication
    for (int k = w; k < nz + w; k++)
    {
        for (int dj = 0; dj < w; dj++)
        {
            // each block length is nx*F, as we can send/recv an entire row of x dimension for all F fields in a contiguous manner in the original data grid.
            blen_y[c] = nx * F;
            // For south face send, the starting point is at y=w, because 0 to w-1 are ghost cells.
            disp_y_ss[c] = IDX(w, w + dj, k, 0);
            // For south face recv, the starting point is at y=0, to copy into the ghost cells on south face
            disp_y_rs[c] = IDX(w, 0 + dj, k, 0);
            // For north face send, the starting point is at y=ny, because ny to ny+w-1 are the ghost cells on north face.
            disp_y_sn[c] = IDX(w, ny + dj, k, 0);
            // For north face recv, the starting point is at y=ny+w, to copy into the ghost cells on north face
            disp_y_rn[c] = IDX(w, ny + w + dj, k, 0);
            // Incrementing the iterator for block length and displacement arrays
            c++;
        }
    }
    // Creating MPI indexed datatypes for y dimension communication using the block length and displacement arrays and committing them
    MPI_Type_indexed(count_y, blen_y, disp_y_ss, MPI_DOUBLE, &type_y_send_south);
    MPI_Type_commit(&type_y_send_south);
    MPI_Type_indexed(count_y, blen_y, disp_y_rs, MPI_DOUBLE, &type_y_recv_south);
    MPI_Type_commit(&type_y_recv_south);
    MPI_Type_indexed(count_y, blen_y, disp_y_sn, MPI_DOUBLE, &type_y_send_north);
    MPI_Type_commit(&type_y_send_north);
    MPI_Type_indexed(count_y, blen_y, disp_y_rn, MPI_DOUBLE, &type_y_recv_north);
    MPI_Type_commit(&type_y_recv_north);

    // Similarly, creating MPI indexed datatypes for z dimension communication.
    MPI_Datatype type_z_send_back, type_z_recv_back, type_z_send_front, type_z_recv_front;
    // For z dimension, we need nx*ny*w*F cells for each face, so the count for block length and displacement arrays will be ny*w,
    // and each block length will be nx*F as we can send/recv an entire row of x dimension for all F fields in a contiguous manner in the original data grid.
    int count_z = ny * w;
    // Allocating block length and displacement arrays for the 4 faces in z dimension (back and front send and receive)
    int *blen_z = malloc(count_z * sizeof(int));
    int *disp_z_sb = malloc(count_z * sizeof(int));
    int *disp_z_rb = malloc(count_z * sizeof(int));
    int *disp_z_sf = malloc(count_z * sizeof(int));
    int *disp_z_rf = malloc(count_z * sizeof(int));

    // Iterator for the block length and displacement arrays
    c = 0;
    // Looping over the z and y dimensions to fill the block length and displacement arrays for z dimension communication
    for (int dk = 0; dk < w; dk++)
    {
        for (int j = w; j < ny + w; j++)
        {
            // each block length is nx*F, as we can send/recv an entire row of x dimension for all F fields in a contiguous manner in the original data grid.
            blen_z[c] = nx * F;
            // For back face send, the starting point is at z=w, because 0 to w-1 are ghost cells.
            disp_z_sb[c] = IDX(w, j, w + dk, 0);
            // For back face recv, the starting point is at z=0, to copy into the ghost cells on back face
            disp_z_rb[c] = IDX(w, j, 0 + dk, 0);
            // For front face send, the starting point is at z=nz, because nz to nz+w-1 are the ghost cells on front face.
            disp_z_sf[c] = IDX(w, j, nz + dk, 0);
            // For front face recv, the starting point is at z=nz+w, to copy into the ghost cells on front face
            disp_z_rf[c] = IDX(w, j, nz + w + dk, 0);
            // Incrementing the iterator for block length and displacement arrays
            c++;
        }
    }
    // Creating MPI indexed datatypes for z dimension communication using the block length and displacement arrays and committing them
    MPI_Type_indexed(count_z, blen_z, disp_z_sb, MPI_DOUBLE, &type_z_send_back);
    MPI_Type_commit(&type_z_send_back);
    MPI_Type_indexed(count_z, blen_z, disp_z_rb, MPI_DOUBLE, &type_z_recv_back);
    MPI_Type_commit(&type_z_recv_back);
    MPI_Type_indexed(count_z, blen_z, disp_z_sf, MPI_DOUBLE, &type_z_send_front);
    MPI_Type_commit(&type_z_send_front);
    MPI_Type_indexed(count_z, blen_z, disp_z_rf, MPI_DOUBLE, &type_z_recv_front);
    MPI_Type_commit(&type_z_recv_front);

    // Allocating temporary buffer for keeping isovalue counts for each time step and field
    long long *local_counts = calloc(T * F, sizeof(long long));
    // Allocating temporary buffer for receiving global counts from all processes to be reduced to rank 0 at the end
    long long *global_counts = calloc(T * F, sizeof(long long));

    // Ensuring all processes have reached this point before starting the timer, to accurately count the timings
    MPI_Barrier(MPI_COMM_WORLD);

    // Starting the timer for the main time loop
    double start_time = MPI_Wtime();

    // --- 6. MAIN TIME LOOP ---
    // For each time step, 0<=t<=T, perform Halo exchange always, but skip isovalue count for t=0, as for the first time only stencil computation should happen
    // The loop runs T+1 times to compute isovalue counts for T steps. However, the stencil computation is not done for the last iteration (t=T)
    // Because there is no next iteration and stencil computation is the last step, preparing stencil for the next iteration, which is not required for t=T.
    // General execution order is Halo exchange, isovalue counting, and stencil computation.
    for (int t = 0; t <= T; t++)
    {

        // 1. ONE-GO HALO EXCHANGE

        // Declaring request array for non-blocking communications, to track each send/recvs
        MPI_Request reqs[12];

        // Initializing request counter to 0 before starting the halo exchange for the current time step
        int req_cnt = 0;

        // If the west neighbor exists,
        if (rank_west != MPI_PROC_NULL)
        {
            // Non-blocking Isend call to send data to west neighbor and incrementing the request counter for current communication
            MPI_Isend(data_current, 1, type_x_send_west, rank_west, 0, MPI_COMM_WORLD, &reqs[req_cnt++]);
            // Non-blocking Irecv call to receive data from west neighbor and incrementing request counter for current communication
            MPI_Irecv(data_current, 1, type_x_recv_west, rank_west, 1, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        // If the east neighbor exists,
        if (rank_east != MPI_PROC_NULL)
        {
            // Non-blocking Isend call to send data to east neighbor and incrementing the request counter for current communication
            MPI_Isend(data_current, 1, type_x_send_east, rank_east, 1, MPI_COMM_WORLD, &reqs[req_cnt++]);
            // Non-blocking Irecv call to receive data from east neighbor and incrementing request counter for current communication
            MPI_Irecv(data_current, 1, type_x_recv_east, rank_east, 0, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        // If the south neighbor exists,
        if (rank_south != MPI_PROC_NULL)
        {
            // Non-blocking Isend call to send data to south neighbor and incrementing the request counter for current communication
            MPI_Isend(data_current, 1, type_y_send_south, rank_south, 2, MPI_COMM_WORLD, &reqs[req_cnt++]);
            // Non-blocking Irecv call to receive data from south neighbor and incrementing request counter for current communication
            MPI_Irecv(data_current, 1, type_y_recv_south, rank_south, 3, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        // If the north neighbor exists,
        if (rank_north != MPI_PROC_NULL)
        {
            // Non-blocking Isend call to send data to north neighbor and incrementing the request counter for current communication
            MPI_Isend(data_current, 1, type_y_send_north, rank_north, 3, MPI_COMM_WORLD, &reqs[req_cnt++]);
            // Non-blocking Irecv call to receive data from north neighbor and incrementing request counter for current communication
            MPI_Irecv(data_current, 1, type_y_recv_north, rank_north, 2, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        // If the back neighbor exists,
        if (rank_back != MPI_PROC_NULL)
        {
            // Non-blocking Isend call to send data to back neighbor and incrementing the request counter for current communication
            MPI_Isend(data_current, 1, type_z_send_back, rank_back, 4, MPI_COMM_WORLD, &reqs[req_cnt++]);
            // Non-blocking Irecv call to receive data from back neighbor and incrementing request counter for current communication
            MPI_Irecv(data_current, 1, type_z_recv_back, rank_back, 5, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }
        // If the front neighbor exists,
        if (rank_front != MPI_PROC_NULL)
        {
            // Non-blocking Isend call to send data to front neighbor and incrementing the request counter for current communication
            MPI_Isend(data_current, 1, type_z_send_front, rank_front, 5, MPI_COMM_WORLD, &reqs[req_cnt++]);
            // Non-blocking Irecv call to receive data from front neighbor and incrementing request counter for current communication
            MPI_Irecv(data_current, 1, type_z_recv_front, rank_front, 4, MPI_COMM_WORLD, &reqs[req_cnt++]);
        }

        // Waiting for all non-blocking communications to complete before proceeding.
        // The request counter currently contains the total number of non-blocking send and receive calls made for the current time step.
        MPI_Waitall(req_cnt, reqs, MPI_STATUSES_IGNORE);

        // Count isovalues only if t>0, for t=0, we compute the stencil first as a part of preprocessing the randomly initialized grid.
        if (t > 0)
        {
            // setting t to t-1, as the isovalue local counts array is 0 indexed, so setting t accordingly.
            // Will be adjusted later to the correct t value at the end of the loop.
            t--;
            // 2. ISOVALUE EDGES COUNTING
            // Looping over the interior points of the local grid (excluding ghost cells) to count the number of edges crossing the isovalue for each field
            // Each dimension starts from w and goes till t+w, where t is any of nx, ny, nz, to cosnider the original grid(excluding ghost cells).
            for (int k = w; k < nz + w; k++)
            {
                for (int j = w; j < ny + w; j++)
                {
                    for (int i = w; i < nx + w; i++)
                    {
                        for (int f = 0; f < F; f++)
                        {

                            // The value at the current grid point for the current field is stored in val
                            double val = data_current[IDX(i, j, k, f)];

                            // Forward Edges (Guarded against global boundary)
                            // X boundary check: my_x*nx+(i-w) is the global x coordinate of the current point, if it is less than px*nx, then we can check the edge in +x direction
                            if (my_x * nx + (i - w) + 1 < px * nx)
                            {
                                // The value at the neighboring point in +x direction for the current field is stored in val_e
                                double val_e = data_current[IDX(i + 1, j, k, f)];
                                // If there is a crossing of the isovalue between the current point and the neighboring point in +x direction
                                if ((val <= isoval && val_e > isoval) || (val > isoval && val_e <= isoval))
                                    // increment the local count for the current time step and field
                                    local_counts[t * F + f]++;
                            }
                            // Y boundary check: my_y*ny+(j-w) is the global y coordinate of the current point, if it is less than py*ny, then we can check the edge in +y direction
                            if (my_y * ny + (j - w) + 1 < py * ny)
                            {
                                // The value at the neighboring point in +y direction for the current field is stored in val_n
                                double val_n = data_current[IDX(i, j + 1, k, f)];
                                // If there is a crossing of the isovalue between the current point and the neighboring point in +y direction
                                if ((val <= isoval && val_n > isoval) || (val > isoval && val_n <= isoval))
                                    // increment the local count for the current time step and field
                                    local_counts[t * F + f]++;
                            }
                            // Z boundary check: my_z*nz+(k-w) is the global z coordinate of the current point, if it is less than pz*nz, then we can check the edge in +z direction
                            if (my_z * nz + (k - w) + 1 < pz * nz)
                            {
                                // The value at the neighboring point in +z direction for the current field is stored in val_f
                                double val_f = data_current[IDX(i, j, k + 1, f)];
                                // If there is a crossing of the isovalue between the current point and the neighboring point in +z direction
                                if ((val <= isoval && val_f > isoval) || (val > isoval && val_f <= isoval))
                                    // increment the local count for the current time step and field
                                    local_counts[t * F + f]++;
                            }
                        }
                    }
                }
            }
            // Adjusting t back to the correct value for the current time step, as it was decremented at the start of this block to access the correct index in local_counts array
            t++;
        }
        // Stencil computation is not done for the last iteration (t=T), because there is no next iteration and stencil computation is the last step, which prepares the grid for the next iteration.
        if (t < T)
        {
            // 3. GENERALIZED STENCIL COMPUTATION
            // Looping over the interior points of the local grid (excluding ghost cells) to compute the new values for the next time step
            for (int k = w; k < nz + w; k++)
            {
                for (int j = w; j < ny + w; j++)
                {
                    for (int i = w; i < nx + w; i++)
                    {
                        for (int f = 0; f < F; f++)
                        {
                            // Initializing the sum as the value of the point itself, from the stencil
                            double sum = data_current[IDX(i, j, k, f)];
                            // Initializing the valid count as 1, to account for the point itself, which is always counted in the stencil
                            int valid = 1;

                            // Looping over the stencil width from 1 to w to accumulate the values of the neighboring points
                            // Compute the neighboring point, i/j/k - w is making the offset 0 indexed. Further, +/- step is to check the neighbor position at step distance.
                            for (int step = 1; step <= w; step++)
                            {
                                // For x direction, checking if the global x coordinate of the neighboring point in -x direction (my_x*nx+(i-w)-step) is >=0
                                if (my_x * nx + (i - w) - step >= 0)
                                {
                                    // If valid, accumulate the value from the neighboring point into sum
                                    sum += data_current[IDX(i - step, j, k, f)];
                                    // Increment the valid count for each valid neighboring point
                                    valid++;
                                }
                                // For x direction, checking if the global x coordinate of the neighboring point in +x direction (my_x*nx+(i-w)+step) is < px*nx
                                if (my_x * nx + (i - w) + step < px * nx)
                                {
                                    // If valid, accumulate the value from the neighboring point into sum
                                    sum += data_current[IDX(i + step, j, k, f)];
                                    // Increment the valid count for each valid neighboring point
                                    valid++;
                                }
                                // For y direction, checking if the global y coordinate of the neighboring point in -y direction (my_y*ny+(j-w)-step) is >=0
                                if (my_y * ny + (j - w) - step >= 0)
                                {
                                    // If valid, accumulate the value from the neighboring point into sum
                                    sum += data_current[IDX(i, j - step, k, f)];
                                    // Increment the valid count for each valid neighboring point
                                    valid++;
                                }
                                // For y direction, checking if the global y coordinate of the neighboring point in +y direction (my_y*ny+(j-w)+step) is < py*ny
                                if (my_y * ny + (j - w) + step < py * ny)
                                {
                                    // If valid, accumulate the value from the neighboring point into sum
                                    sum += data_current[IDX(i, j + step, k, f)];
                                    // Increment the valid count for each valid neighboring point
                                    valid++;
                                }
                                // For z direction, checking if the global z coordinate of the neighboring point in -z direction (my_z*nz+(k-w)-step) is >=0
                                if (my_z * nz + (k - w) - step >= 0)
                                {
                                    // If valid, accumulate the value from the neighboring point into sum
                                    sum += data_current[IDX(i, j, k - step, f)];
                                    // Increment the valid count for each valid neighboring point
                                    valid++;
                                }
                                // For z direction, checking if the global z coordinate of the neighboring point in +z direction (my_z*nz+(k-w)+step) is < pz*nz
                                if (my_z * nz + (k - w) + step < pz * nz)
                                {
                                    // If valid, accumulate the value from the neighboring point into sum
                                    sum += data_current[IDX(i, j, k + step, f)];
                                    // Increment the valid count for each valid neighboring point
                                    valid++;
                                }
                            }
                            // Compute the average by dividing the sum by the valid count and store it in data_next for the next time step
                            data_next[IDX(i, j, k, f)] = sum / (double)valid;
                        }
                    }
                }
            }

            // 4. POINTER SWAP
            // Swapping the pointers of data_current and data_next to avoid copying the entire grid for the next iteration.
            double *temp = data_current;
            data_current = data_next;
            data_next = temp;
        }
    }

    // Stopping the timer after the main time loop has completed
    double end_time = MPI_Wtime();
    // Calculating the total time taken for the main time loop by taking the difference of end time and start time
    double total_time = end_time - start_time;
    // Variable to store the maximum time taken across all processes, which will be calculated in the reduction step
    double max_time;

    // --- 7. REDUCTIONS AND OUTPUT ---
    // Reducing the total time taken across all processes to find the maximum time
    MPI_Reduce(&total_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    // Reducing the local isovalue counts across all processes to get the global counts for each time step and field
    MPI_Reduce(local_counts, global_counts, T * F, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    // Printing the global isovalue counts and maximum time by rank 0
    if (myrank == 0)
    {
        // For each time step
        for (int t = 0; t < T; t++)
        {
            // For each field
            for (int f = 0; f < F; f++)
            {
                // Printing the global count for current time step and field
                printf("%lld ", global_counts[t * F + f]);
            }
            printf("\n");
        }
        // Printing the maximum time taken across all processes for the main time loop
        printf("%f\n", max_time);
    }

    // --- 8. CLEANUP ---
    // Freeing the MPI datatypes created for halo exchange
    MPI_Type_free(&type_x_send_west);
    MPI_Type_free(&type_x_recv_west);
    MPI_Type_free(&type_x_send_east);
    MPI_Type_free(&type_x_recv_east);
    MPI_Type_free(&type_y_send_south);
    MPI_Type_free(&type_y_recv_south);
    MPI_Type_free(&type_y_send_north);
    MPI_Type_free(&type_y_recv_north);
    MPI_Type_free(&type_z_send_back);
    MPI_Type_free(&type_z_recv_back);
    MPI_Type_free(&type_z_send_front);
    MPI_Type_free(&type_z_recv_front);

    // Freeing the block length and displacement arrays used for creating MPI datatypes
    free(blen_x);
    free(disp_x_sw);
    free(disp_x_rw);
    free(disp_x_se);
    free(disp_x_re);
    free(blen_y);
    free(disp_y_ss);
    free(disp_y_rs);
    free(disp_y_sn);
    free(disp_y_rn);
    free(blen_z);
    free(disp_z_sb);
    free(disp_z_rb);
    free(disp_z_sf);
    free(disp_z_rf);

    // Freeing the data arrays for stencil computation
    free(data_current);
    free(data_next);

    // Freeing the local and global counts arrays used for isovalue counting
    free(local_counts);
    free(global_counts);

    // Finalizing the MPI environment before exiting the program
    MPI_Finalize();

    // Returning from main function
    return 0;
}

