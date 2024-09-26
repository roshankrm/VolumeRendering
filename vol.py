import numpy as np
from mpi4py import MPI
import sys
from PIL import Image
import time
import bisect    
def generate_sub_image(sub_data,step_size,color_dict,opacity_dict):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    rows,cols,depth=sub_data.shape
    sub_image = np.zeros((rows, cols, 3), dtype=np.float32)
    sub_opacity = np.zeros((rows, cols), dtype=np.float32)
    early_ray_terminations=0
    
    colors = list(color_dict.items())
    opacities = list(opacity_dict.items())
    color_keys = np.array(sorted(color_dict.keys()), dtype=np.float64)
    opacity_keys = np.array(sorted(opacity_dict.keys()), dtype=np.float64)
    
    for i in range(rows):
        for j in range(cols):
            sample_point=0
            
            while(sample_point<depth-1):
                before_index=int(np.floor(sample_point))
                next_index=before_index+1
                alpha=sample_point-before_index
                sample_point_value=(1-alpha)*sub_data[i,j,before_index]+alpha*sub_data[i,j,next_index]
                col=[0,0,0]
                
                index = bisect.bisect(color_keys, sample_point_value)
                if index==54:
                    index=53
                width = color_keys[index] - color_keys[index-1]
                distance_to_left = sample_point_value - color_keys[index-1]
                alpha = distance_to_left / width

                
                
                col[0] = (1 - alpha) * colors[index-1][1][0] + alpha * colors[index][1][0]
                col[1] = (1 - alpha) * colors[index-1][1][1] + alpha * colors[index][1][1]
                col[2] = (1 - alpha) * colors[index-1][1][2] + alpha * colors[index][1][2]

                index_dict=bisect.bisect(opacity_keys, sample_point_value)
                if index_dict==4:
                    index_dict=3
                width = opacity_keys[index_dict] - opacity_keys[index_dict-1]
                distance_to_left = sample_point_value - opacity_keys[index_dict-1]
                alpha = distance_to_left / width

                
                
                opa=(1 - alpha) * opacities[index_dict-1][1] + alpha * opacities[index_dict][1]
                
                
                opacity=sub_opacity[i, j]
                one_minus_opacity = 1.0 - opacity
                sub_image[i, j, 0] += col[0] * one_minus_opacity * opa
                sub_image[i, j, 1] += col[1] * one_minus_opacity * opa
                sub_image[i, j, 2] += col[2] * one_minus_opacity * opa
                sub_opacity[i, j] +=  one_minus_opacity*opa
                
                if (sub_opacity[i, j]==1):
                    early_ray_terminations+=1
                    break

                sample_point+=step_size

    return np.ascontiguousarray(sub_image),sub_opacity,early_ray_terminations



def opacity_function(filename):
    """
    Converts the contents of a file into a dictionary where each odd line
    is a key and the subsequent even line is the corresponding value.
    :param filename: The name of the file to read from.
    :return: A dictionary with keys and values extracted from the file.
    """
    opacity_dict = {}

    with open(filename, 'r') as file:
        lines = file.readlines()

    for i in range(0, len(lines), 2):
        key = lines[i].strip()
        key=float(key[:-1])
        value = lines[i + 1].strip()
        if value[-1]==",":
            opacity_dict[key] = float(value[:-1])
        else :
            opacity_dict[key]=float(value)

    return opacity_dict

def color_function(filename):
    """
    Converts the contents of a file into a dictionary where each first line
    in every group of four lines is a key, and the next three lines form a list as the corresponding value.

    :param filename: The name of the file to read from.
    :return: A dictionary with keys and list values extracted from the file.
    """
    color_dict = {}

    with open(filename, 'r') as file:
        lines = file.readlines()


    i = 0
    while i < len(lines):
        key = lines[i].strip() 
        key =float( key.rstrip(':')[:-1] )

   
        if i + 3 < len(lines):
            value = list(lines[i + 1:i + 4])
            value = list(map(str.strip, value))  

            
            for j in range(3):
                if value[j].endswith(","):
                    value[j] = float(value[j][:-1])
                else:
                    value[j]=float(value[j])

            color_dict[key] =value
            i += 4  
        else:
            break 

    return color_dict


def read_data(filename, z_dim, y_dim, x_dim,x_min,x_max,y_min,y_max):
    data = np.fromfile(filename, dtype=np.float32)
    data = data.reshape((z_dim, y_dim, x_dim))
    data=np.transpose(data, (2, 1, 0))
    data=data[x_min:x_max+1,y_min:y_max+1,:]
    return np.ascontiguousarray(data)
    
def find_best_split(num_processes):
    best_split = (1, num_processes)
    for i in range(1, int(np.sqrt(num_processes)) + 1):
        if num_processes % i == 0:
            j = num_processes // i
            if i <= j:
                best_split = (j, i)
    return best_split
    
def divide_data_2d(data, num_processes, rank):
    x_dim, y_dim, z_dim = data.shape
    

    num_splits_x, num_splits_y = find_best_split(num_processes)

    chunk_x = x_dim // num_splits_x
    chunk_y = y_dim // num_splits_y

    proc_x = rank % num_splits_x
    proc_y = rank // num_splits_x

    start_x = proc_x * chunk_x
    end_x = x_dim if proc_x == num_splits_x - 1 else start_x + chunk_x
    start_y = proc_y * chunk_y
    end_y = y_dim if proc_y == num_splits_y - 1 else start_y + chunk_y

    local_data = data[start_x:end_x, start_y:end_y, :]
    return np.ascontiguousarray(local_data),start_x,end_x,start_y,end_y


def divide_data_1d(data, num_processes, rank):
    x_dim, y_dim, z_dim = data.shape
    chunk_size = x_dim // num_processes
    start_x = rank * chunk_size
    end_x = x_dim if rank == num_processes - 1 else start_x + chunk_size

    local_data = data[start_x:end_x, :, :]
    return np.ascontiguousarray(local_data),start_x,end_x


def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    filename = sys.argv[1]
    partitioning = int(sys.argv[2])
    step_size = float(sys.argv[3])
    x_bound_min = int(sys.argv[4])
    x_bound_max = int(sys.argv[5])
    y_bound_min = int(sys.argv[6])
    y_bound_max = int(sys.argv[7])
    x_max=int(filename[7:11])
    y_max=int(filename[12:16])
    z_max=int(filename[17:20])
    op_dictionary = opacity_function("opacity_TF.txt")
    color_dictionary = color_function("color_TF.txt")
    
    if rank == 0:
        final_ans1 = np.zeros((x_bound_max - x_bound_min + 1, y_bound_max - y_bound_min + 1,4))
        final_ans=np.ascontiguousarray(final_ans1)
        max_time=-1
        START_OF_Xs=[]
        END_OF_Xs=[]
        START_OF_Ys=[]
        END_OF_Ys=[]
        COUNT_OF_TERMINATIONS=0
        z_dim = z_max
        y_dim = y_max
        x_dim = x_max
        data = read_data(filename, z_dim, y_dim, x_dim, x_bound_min, x_bound_max, y_bound_min, y_bound_max)
        op_dictionary = opacity_function("opacity_TF.txt")
        color_dictionary = color_function("color_TF.txt")
        

        sub_domain_0=np.array([1])
        if partitioning == 1:
            for i in range(0,size):
                process_data, s_x, e_x = divide_data_1d(data, size, i)
                START_OF_Xs.append(s_x)
                END_OF_Xs.append(e_x)
                START_OF_Ys.append(0)
                END_OF_Ys.append(y_bound_max - y_bound_min+1)
                if i!=0:
                    comm.Send(process_data, dest=i)  
                else:
                    sub_domain_0=np.copy(process_data)
        else:
            for i in range(0,size):
                process_data, s_x, e_x, s_y, e_y = divide_data_2d(data, size, i)
                START_OF_Xs.append(s_x)
                END_OF_Xs.append(e_x)
                START_OF_Ys.append(s_y)
                END_OF_Ys.append(e_y)
                if i!=0: 
                    comm.Send(process_data, dest=i) 
                else:
                    sub_domain_0=np.copy(process_data)
        
        top_x=START_OF_Xs[0]
        bottom_x=END_OF_Xs[0]
        left_y=START_OF_Ys[0]
        right_y=END_OF_Ys[0] 
        start_time=time.time()
        sub_image,sub_opacity,early_ray_terminations=generate_sub_image(sub_domain_0,step_size,color_dictionary,op_dictionary)
        end_time=time.time()
        print("process 0 time ", end_time-start_time)
        max_time=max(max_time,end_time-start_time)
        rgba_image = np.dstack((sub_image, sub_opacity))
        final_ans[top_x:bottom_x,left_y:right_y]=rgba_image
        COUNT_OF_TERMINATIONS+=early_ray_terminations 
        
        for i in range(1,size):
            top_x=START_OF_Xs[i]
            bottom_x=END_OF_Xs[i] 
            left_y=START_OF_Ys[i]
            right_y=END_OF_Ys[i] 
            receiving_subarray = np.zeros((bottom_x - top_x, right_y - left_y, 4), dtype=np.float32)
            comm.Recv(receiving_subarray,source=i)
            curr_early_terminations=comm.recv(source=i)
            curr_time=comm.recv(source=i)
            print("process ",i,"time",curr_time)
            max_time=max(max_time,curr_time)
            COUNT_OF_TERMINATIONS+=curr_early_terminations
            final_ans[top_x:bottom_x,left_y:right_y]=np.copy(receiving_subarray);    

        print("No of terminations:",COUNT_OF_TERMINATIONS)
        Fraction_of_terminations=COUNT_OF_TERMINATIONS/((y_bound_max-y_bound_min+1)*(x_bound_max-x_bound_min+1))
        print("Fraction of terminations",Fraction_of_terminations)
        print("Max time taken is:",max_time)
        image = Image.fromarray((final_ans * 255).astype(np.uint8), 'RGBA')
        image.save('output_image.png')
            
    else:
        if partitioning == 1:
            x_dim, y_dim, z_dim = x_bound_max - x_bound_min + 1, y_bound_max - y_bound_min + 1, z_max
            chunk_size = x_dim // size
            start_x = rank * chunk_size
            end_x = x_dim if rank == size - 1 else start_x + chunk_size
            subdomain_shape = (end_x - start_x, y_dim, z_dim)
            subdomain = np.empty(subdomain_shape, dtype=np.float32)
            comm.Recv(subdomain, source=0)
            start_time=time.time()
            sub_image,sub_opacity,early_ray_terminations=generate_sub_image(subdomain,step_size,color_dictionary,op_dictionary)
            end_time=time.time()
            rgba_image = np.dstack((sub_image, sub_opacity))
            comm.Send(rgba_image,dest=0)
            comm.send(early_ray_terminations,dest=0)
            comm.send(end_time-start_time,dest=0)
            
        elif partitioning == 2:
            x_dim, y_dim, z_dim = x_bound_max - x_bound_min + 1, y_bound_max - y_bound_min + 1, z_max
            num_splits_x, num_splits_y = find_best_split(size)
            chunk_x = x_dim // num_splits_x
            chunk_y = y_dim // num_splits_y

            proc_x = rank % num_splits_x
            proc_y = rank // num_splits_x

            start_x = proc_x * chunk_x
            end_x = x_dim if proc_x == num_splits_x - 1 else start_x + chunk_x
            start_y = proc_y * chunk_y
            end_y = y_dim if proc_y == num_splits_y - 1 else start_y + chunk_y

            subdomain_shape = (end_x - start_x, end_y - start_y, z_dim)
            subdomain = np.empty(subdomain_shape, dtype=np.float32)
            comm.Recv(subdomain, source=0)
            start_time=time.time()
            sub_image,sub_opacity,early_ray_terminations=generate_sub_image(subdomain,step_size,color_dictionary,op_dictionary)
            end_time=time.time()
            rgba_image = np.dstack((sub_image, sub_opacity))
            comm.Send(rgba_image,dest=0)
            comm.send(early_ray_terminations,dest=0)
            comm.send(end_time-start_time,dest=0)
            
            
if __name__ == "__main__":
    main()
    
