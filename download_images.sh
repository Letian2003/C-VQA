# download images for C-VQA-Real and C-VQA-Synthetic

gdown https://drive.google.com/uc?id=19WXgtBVYuv_JO4HdOzyEuHGlCtOnvOaf -O C-VQA/C-VQA-Real/C-VQA-Real_images.zip
unzip -o C-VQA/C-VQA-Real/C-VQA-Real_images.zip -d C-VQA/C-VQA-Real/

gdown https://drive.google.com/uc?id=1DsZP_dV9yAyYAo2nB8O0ewD-nTmefnaR -O C-VQA/C-VQA-Synthetic/C-VQA-Synthetic_images.zip
unzip -o C-VQA/C-VQA-Synthetic/C-VQA-Synthetic_images.zip -d C-VQA/C-VQA-Synthetic/

rm -rf C-VQA/C-VQA-Real/C-VQA-Real_images.zip
rm -rf C-VQA/C-VQA-Synthetic/C-VQA-Synthetic_images.zip