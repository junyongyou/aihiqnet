# Databases for training PHIQnet

This work uses two publicly available databases: KonIQ-10k [KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment](https://ieeexplore.ieee.org/document/8968750) by V. Hosu, H. Lin, T. Sziranyi, and D. Saupe;
 and SPAQ [Perceptual quality assessment of smartphone photography](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html) by Y. Fang, H. Zhu, Y. Zeng, K. Ma, and Z. Wang.

The koniq10k_images_scores.csv and image_mos_spaq.csv list all the image names and MOS values (for KonIQ-10k: distribution and MOS in [1, 5]; for SPAQ: MOS in [0, 100]), which were randomly chosen from the two databases in terms of SI (spatial perceptual information) and MOS.

Please define the image_score file in the followig formats:
    For database with score distribution available, the MOS file is like this :
    ```
        image path, voter number of quality scale 1, voter number of quality scale 2, voter number of quality scale 3, voter number of quality scale 4, voter number of quality scale 5, MOS or Z-score
        image1.jpg;0,0,25,73,7,3.828571429
        image2.jpg;0,3,45,47,1,3.479166667
        image3.jpg;1,0,20,73,2,3.78125
        image4.jpg;0,0,21,75,13,3.926605505
    ```

    For database with MOS values only, the MOS file is like this (SPAQ format):
    ```
        image path, MOS or Z-score
        image1.jpg;3.828571429
        image2.jpg;3.479166667
        image3.jpg;2,3.78125
        image4.jpg;3.926605505
    ```

Please split train/val/test sets appropriately. If the absolute path of image is not specified in the image_score file, then it should be defined in args['image_folder']

train_val_test_koniq.pkl and train_val_test_spaq.pkl give two examples of random split of train/val/test sets. They are dumped by pickle, so if the train/val/test sets are stored in another way, please change lines 173, 174, 181 and 182 in train.py.