# Databases for training PHIQnet

This work uses two publicly available databases: KonIQ-10k [KonIQ-10k: An ecologically valid database for deep learning of blind image quality assessment](https://ieeexplore.ieee.org/document/8968750) by V. Hosu, H. Lin, T. Sziranyi, and D. Saupe;
 and LIVE-wild [Perceptual quality assessment of smartphone photography](https://openaccess.thecvf.com/content_CVPR_2020/html/Fang_Perceptual_Quality_Assessment_of_Smartphone_Photography_CVPR_2020_paper.html) by Y. Fang, H. Zhu, Y. Zeng, K. Ma, and Z. Wang.

The koniq10k_images_scores.csv and image_mos_spaq.csv list all the image names and MOS values (for KonIQ-10k: distribution and MOS in [1, 5]; for SPAQ: MOS in [0, 100]), which were randomly chosen from the two databases in terms of SI (spatial perceptual information) and MOS.

train_val_test_koniq.pkl and train_val_test_spaq.pkl give two examples of random split of train/val/test sets.