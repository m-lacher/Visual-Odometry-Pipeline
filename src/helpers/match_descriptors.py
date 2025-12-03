import numpy as np
from scipy.spatial.distance import cdist

def matchDescriptorsLOWE(img1_descriptors, img0_descriptors, match_lambda):
    # img0 = target, img1 = candidate
    
    print("Query descriptor:", img1_descriptors.shape)
    print("Target descriptor:", img0_descriptors.shape)

    # computing distances between descriptors
    # disntances is a QxD matrix where Q is the number of query descriptors and D is the number of database descriptors. 
    # In our case its 200 by 200 because both images have 200 keypoints
    # We will probably have to think about the number of keypoints later for performance reasons 200 seems like a lot

    distances = cdist(img1_descriptors.T, img0_descriptors.T, 'euclidean')
    #print(distances.shape)
    #print(distances[0][:])
    sorted_indices = np.argsort(distances, axis=1)
    
    #print(sorted_indices[0][:])
    # This shows the list of which keypoint in the database is the closest to keypoint 0 in the query
    
    matches = np.ones(distances.shape[0], dtype=int) * -1
    
    # this filters out ambiguous matches (lowe ratio)
    # we iterate through each descriptor, if the first and second are not close than we store the best match index in the matches array otherwise that entry is a -1 for that match
    for i in range(distances.shape[0]):
        best_idx = sorted_indices[i, 0]
        second_best_idx = sorted_indices[i, 1]
        
        dist1 = distances[i, best_idx]
        dist2 = distances[i, second_best_idx]
        
        if dist1 < match_lambda * dist2:
            matches[i] = best_idx

    descriptor_count = distances.shape[1]
    for img0_idx in range(descriptor_count):
        # here we find all the indices from the img1 descriptors that match to the current img0 descriptor
        matching_img1_descriptors = []
        for i in range(len(matches)):
            if matches[i] == img0_idx:
                matching_img1_descriptors.append(i)
        
        # two or more descriptors machting
        if len(matching_img1_descriptors) > 1:
            
            minimum_distance = float('inf')
            winning_query_index = -1
            
            for img1_idx in matching_img1_descriptors:
                # checking actual distance by querying the distances matrix calculated at the beginning
                current_distance = distances[img1_idx, img0_idx]
                
                if current_distance < minimum_distance:
                    # overwrite because a closer match found
                    print("closer match found overwriting")
                    minimum_distance = current_distance
                    winning_query_index = img1_idx
            
            # losing the non-best matches
            matches[matching_img1_descriptors] = -1
            # claiming the index by the best match
            matches[winning_query_index] = img0_idx

    return matches
