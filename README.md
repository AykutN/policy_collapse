# Multi-Objective Reinforcement Learning (MORL) - HalfCheetah

Bu proje, HalfCheetah ortamında Çok Amaçlı Pekiştirmeli Öğrenme (Multi-Objective Reinforcement Learning, MORL) algoritmalarını implement eden kapsamlı bir framework'tür. PPO (Proximal Policy Optimization) ve GRPO (Gradient Projection-based Optimization) algoritmaları kullanarak çoklu objektif optimizasyonu gerçekleştirir.

## 🚀 Özellikler

- **HalfCheetah MORL Ortamı**: Özelleştirilmiş 3 amaçlı (hız, enerji verimliliği, hareket yumuşaklığı) ortam
- **PPO + GRPO Algoritması**: Grup tabanlı gradient projection ile geliştirilmiş PPO
- **Tercih Örnekleme**: Dirichlet dağılımı ve köşe noktaları ile tercih vektörü üretimi
- **Kapsamlı Metrikler**: 
  - Hypervolume (Monte Carlo & Exact)
  - IGD (Inverted Generational Distance)
  - IGD+ metrikleri
- **Görselleştirme**: UMAP/t-SNE ile embedding manifold analizi
- **Pareto Front Analizi**: Çok amaçlı optimizasyon sonuçlarının analizi
- **Toplu Deneyim**: Çoklu seed ve varyant ile otomatik deneyim yürütme

## 📋 Gereksinimler

### Temel Kütüphaneler
```
gymnasium
mujoco
pymoo
numpy
pandas
matplotlib
seaborn
scikit-learn
umap-learn
python-ternary
```

## 🛠 Kurulum

### 1. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Depoyu klonlayın:
```bash
git clone https://github.com/AykutN/policy_collapse.git
cd policy_collapse
```

## 📖 Kullanım

### Temel Eğitim

Tek bir deneyim çalıştırmak için:

```bash
python training_morl_example.py --max_episodes 5000 --update_timestep 4000
```

### Gelişmiş Seçenekler

```bash
python training_morl_example.py \
    --max_episodes 10000 \
    --update_timestep 4000 \
    --K_epochs 20 \
    --entropy_coef 0.01 \
    --exact_hv \
    --manifold umap \
    --embed_interval 50 \
    --summary_table
```

### Çoklu Deneyim (Farklı Seed'ler)

```bash
python run_many.py \
    --seeds 0 1 2 3 4 \
    --episodes 5000 \
    --exact_hv \
    --manifold umap \
    --variants grpo no-grpo
```

### Google Colab Kullanımı

Google Colab'da çalıştırmak için `colab_morl_halfcheetah.ipynb` notebook'unu kullanabilirsiniz. Bu notebook otomatik kurulum ve fallback ortam içerir.

## 📊 Analiz ve Görselleştirme

### Temel Grafik Çizimi

```bash
python grafik_cizdir.py --run_dir logs/run_XXXXXX
```

### Gelişmiş Analiz

```bash
python advanced_analysis.py --all --log_dir logs
```

### Embedding Görselleştirme

```bash
python embed_visualize.py --run_dir logs/run_XXXXXX
```

## 📁 Proje Yapısı

```
├── training_morl_example.py    # Ana eğitim scripti
├── morl_env_halfcheetah.py     # HalfCheetah MORL ortam wrapper'ı
├── utils_prefs.py              # Tercih vektörü yardımcı fonksiyonları
├── run_many.py                 # Çoklu deneyim yürütücü
├── grafik_cizdir.py           # Temel grafik çizim araçları
├── advanced_analysis.py        # İleri seviye analiz araçları
├── embed_visualize.py          # Embedding görselleştirme
├── colab_morl_halfcheetah.ipynb # Google Colab notebook'u
└── requirements.txt            # Gerekli kütüphaneler
```

## 🎯 MORL Ortamı Detayları

### HalfCheetah-v4 Amaçları

1. **Hız Objektifi**: Hedef hıza ulaşma/sürdürme
2. **Enerji Verimliliği**: Minimum enerji tüketimi
3. **Hareket Yumuşaklığı**: Aksiyon değişimlerini minimize etme

### Ortam Konfigürasyonu

```python
from morl_env_halfcheetah import make_hc_morl

env = make_hc_morl(
    speed_mode="target_speed",
    target_speed=2.0,
    alpha_energy=0.1,
    beta_smooth=0.05,
    normalize=True,
    friction_rand=True,
    seed=42
)
```

## 📈 Metrikler ve Değerlendirme

### Hypervolume (HV)
- Monte Carlo tahmini (hızlı)
- Exact hesaplama (pymoo, 3 amaç için)

### IGD ve IGD+ Metrikleri
- Referans Pareto front ile karşılaştırma
- Convergence ve diversity ölçümü

### Embedding Analizi
- UMAP/t-SNE ile action-preference space görselleştirme
- Policy manifold analizi

## 🔧 Parametre Ayarları

### Ana Eğitim Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--max_episodes` | 50000 | Maksimum episode sayısı |
| `--update_timestep` | 4000 | Güncelleme aralığı |
| `--K_epochs` | 20 | PPO epoch sayısı |
| `--entropy_coef` | 0.01 | Entropy katsayısı |
| `--exact_hv` | False | Exact hypervolume hesaplama |
| `--manifold` | none | Embedding tipi (umap/tsne/none) |

### GRPO Parametreleri

| Parametre | Varsayılan | Açıklama |
|-----------|------------|----------|
| `--no_grpo` | False | GRPO'yu devre dışı bırak |
| `grpo_group_mode` | knn | Gruplandırma modu |
| `grpo_knn_delta` | 0.15 | KNN delta threshold |

## 📊 Çıktı Dosyaları

### Eğitim Çıktıları
- `logs/run_XXXXXX/metrics.csv` - Eğitim metrikleri
- `logs/run_XXXXXX/summary.csv` - Özet istatistikler
- `logs/run_XXXXXX/final_front.npy` - Son Pareto front

### Analiz Çıktıları
- `logs/run_XXXXXX/analysis/` - Grafik ve analiz dosyaları
- `logs/run_XXXXXX/embeddings/` - UMAP/t-SNE embeddings

### Toplu Analiz
- `logs/aggregate_summary.csv` - Tüm run'ların özeti
- `logs/aggregate_summary_stats.csv` - İstatistiksel özetler

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır.

## 📞 İletişim

Aykut N. - [@AykutN](https://github.com/AykutN)

Proje Linki: [https://github.com/AykutN/policy_collapse](https://github.com/AykutN/policy_collapse)

## 🙏 Teşekkürler

- [Gymnasium](https://gymnasium.farama.org/) - RL ortamları
- [MuJoCo](https://mujoco.org/) - Fizik simülasyonu
- [PyMOO](https://pymoo.org/) - Multi-objective optimization
- [UMAP](https://umap-learn.readthedocs.io/) - Dimensionality reduction