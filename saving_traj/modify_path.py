import numpy as np

if __name__ == "__main__":
    trajFolder = '/root/paths/'
    count = 0
    saving_path_folder = '/root/paths_retry/'
    
    for entry in os.listdir(trajFolder):
        if '.npy' in entry:
            s = int(entry.split(".")[0])
            path_array = []
            traj = np.load(osp.join(trajFolder,entry),allow_pickle=True)
            count = count+1
            for points in traj:
                x,y,z = points.x,points.y,points.z
                path_array.append((x,y,z))
                
            np.save(saving_path_folder + str(s) + '.npy',path_array)
            print(count)