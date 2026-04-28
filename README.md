# Textile-Defect-Detection
# Yapay Zeka Destekli Tekstil Yüzey Kusuru Tespiti

Bu proje, tekstil yüzeylerindeki kusurları yapay zeka (derin öğrenme) teknikleri kullanarak otomatik olarak tespit etmeyi amaçlayan bir bitirme projesidir.

## 👥 Ekip Üyeleri ve Rol Dağılımı
* **Kişi 1 - [Mohammed emin]:** Veri ön işleme, Augmentation, DataLoader, GUI (Streamlit/Gradio)
* **Kişi 2 - [Baraa]:** Model mimarisi (U-Net, YOLO vb.), eğitim, hiperparametre, optimizasyon
* **Kişi 3 - [Abdulrahman Ghazi]:** Git/GitHub altyapısı, metrik hesaplama (IoU, F1 vb.), karşılaştırma, akademik rapor ve sunum

## 🛠 Kullanılan Teknolojiler ve Araçlar
* Python, PyTorch, torchvision
* Albumentations, Matplotlib, Seaborn
* YOLOv8 (Ultralytics), segmentation-models-pytorch
* Scikit-learn, Pandas
* Streamlit / Gradio (GUI için)
* TensorBoard / W&B

## 📌 Git & GitHub Geliştirme Kuralları
Ekip içi uyumu sağlamak için aşağıdaki kurallara uyulmalıdır:
1. **Branch (Dallanma) Stratejisi:** `main` (kararlı sürüm), `dev` (geliştirme), ve `feature-x` (yeni özellikler) dalları kullanılacaktır.
2. **Commit Mesajları:** Anlaşılır ve açıklayıcı olmalıdır. 
   - *Örnek:* `feat: add augmentation pipeline` veya `fix: dataloader bug`.
3. **Dosya Kısıtlamaları:** Büyük model dosyaları (`.pt`, `.onnx`) ve büyük veri setleri `.gitignore` dosyası ile repo dışında tutulmalıdır, GitHub'a yüklenmeyecektir.
4. **Deney Logları:** Tüm model denemeleri `experiments/` klasöründe loglanmalıdır.
