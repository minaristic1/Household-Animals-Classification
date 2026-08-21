# Household Animals Classification

Projekat iz predmeta **Mašinsko učenje** na master akademskim studijama Matematičkog fakulteta Univerziteta u Beogradu.

## Članovi tima

- **Mina Ristić — 1119/2025**
- **Alma Hodžić — 1120/2025**

---

## Opis projekta

Cilj projekta je klasifikacija slika pasa i mačaka primenom metoda dubokog učenja.

Tokom projekta implementirana su i analizirana tri pristupa:

1. **Baseline CNN (VGG-6)** treniran od početka
2. **Transfer Learning sa VGG-16** mrežom prethodno treniranom na ImageNet skupu
3. **Fine-tuning VGG-16** modela

Pored poređenja performansi modela, analizirana je i interpretabilnost neuronskih mreža.

Za vizuelizaciju naučenih karakteristika korišćena su dva pristupa:

- vizualizacija međuslojnih aktivacija,
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**.

Grad-CAM omogućava vizuelizaciju delova slike koji najviše utiču na odluku finalnog modela.

Projekat je inspirisan radom:

**Lin, Household Animals Classification Using Deep Learning, Stanford CS230, Winter 2020.**

---

## Skup podataka

Za projekat je korišćen **Dogs vs. Cats** skup podataka.

Skup sadrži ukupno **25.000 slika**:

- 12.500 slika mačaka
- 12.500 slika pasa

Skup je balansiran između dve klase.

Za sve eksperimente koristi se ista podela podataka definisana fajlom `split.csv`.

| Skup | Broj slika | Mačke | Psi |
|---|---:|---:|---:|
| Train | 20.000 | 10.000 | 10.000 |
| Validation | 4.000 | 2.000 | 2.000 |
| Test | 1.000 | 500 | 500 |
| **Ukupno** | **25.000** | **12.500** | **12.500** |

Podela odgovara odnosu:

- **80% train**
- **16% validation**
- **4% test**

Podela je stratifikovana i koristi se fiksirani random seed:

```python
SEED = 42
```

`split.csv` se kreira u notebook-u `01_data_exploration.ipynb` i koristi u svim narednim notebook-ovima.

Na taj način svi modeli se treniraju i evaluiraju na istoj podeli podataka, što omogućava korektno međusobno poređenje.

Test skup se ne koristi tokom treniranja niti za izbor hiperparametara, već isključivo za finalnu evaluaciju modela.

Dataset nije uključen direktno u GitHub repozitorijum zbog veličine.

Dogs vs. Cats dataset dostupan je na Kaggle platformi:

https://www.kaggle.com/c/dogs-vs-cats

---

## Struktura repozitorijuma

```text
Household-Animals-Classification/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_cnn.ipynb
│   ├── 03_transfer_learning.ipynb
│   ├── 04_finetuning.ipynb
│   ├── 05_visualizations_intermediate_activations.ipynb
│   ├── 05_vizualizations_grad_cam.ipynb
│   └── 06_demo.ipynb
│
├── split.csv
└── README.md
```

Notebook-i su numerisani redosledom kojim je projekat razvijan.

Postoje dva notebook-a sa oznakom `05` zato što se interpretabilnost modela analizira kroz dva različita pristupa:

- `05_visualizations_intermediate_activations.ipynb` — vizualizacija međuslojnih aktivacija,
- `05_vizualizations_grad_cam.ipynb` — Grad-CAM analiza finalnog modela.

---

## 01 — Data Exploration

Notebook `01_data_exploration.ipynb` služi za upoznavanje sa skupom podataka i pripremu zajedničke podele podataka.

U ovoj fazi vrši se:

- pregled skupa podataka,
- analiza klasa `cat` i `dog`,
- provera balansiranosti,
- vizuelizacija primera slika,
- formiranje train, validation i test skupova,
- čuvanje konačne podele u `split.csv`.

Kreirani `split.csv` predstavlja jedinstvenu podelu koja se koristi u svim narednim eksperimentima.

---

## 02 — Baseline model

U notebook-u `02_baseline_cnn.ipynb` implementiran je osnovni konvolutivni model treniran potpuno od početka, bez prethodno naučenih težina.

Model prati jednostavniju VGG-style arhitekturu sa više `Conv2D` i `MaxPooling2D` slojeva.

Osnovna struktura modela je:

```text
Conv2D(32)
↓
MaxPooling2D
↓
Conv2D(64)
↓
MaxPooling2D
↓
Conv2D(128)
↓
MaxPooling2D
↓
Conv2D(128)
↓
MaxPooling2D
↓
Flatten
↓
Dense(512)
↓
Dense(1, sigmoid)
```

Prvo je trenirana verzija modela bez regularizacije kako bi se analizirala pojava overfitting-a.

Nakon toga implementirana je regularizovana verzija korišćenjem:

- horizontalnog preslikavanja,
- nasumične rotacije,
- nasumičnog zumiranja,
- `Dropout(0.5)`,
- `EarlyStopping`.

Finalni baseline predstavlja regularizovanu verziju modela.

### Rezultati baseline modela

| Metrika | Rezultat |
|---|---:|
| Accuracy | **90.30%** |
| Precision | **89.74%** |
| Recall | **91.00%** |
| F1-score | **90.37%** |

Matrica konfuzije:

```text
                 Predicted cat    Predicted dog
Actual cat             448              52
Actual dog              45             455
```

Od ukupno 1.000 test slika pravilno su klasifikovane **903 slike**.

Finalni baseline model čuva se kao:

```text
vgg6_baseline_regularized.keras
```

---

## 03 — Transfer Learning sa VGG-16

U notebook-u `03_transfer_learning.ipynb` primenjen je transfer learning korišćenjem **VGG-16** mreže prethodno trenirane na ImageNet skupu.

VGG-16 se učitava sa:

```python
weights="imagenet"
include_top=False
```

Originalni klasifikacioni deo VGG-16 mreže nije korišćen.

Tokom ove faze cela konvoluciona baza bila je zamrznuta:

```python
base_model.trainable = False
```

Na VGG-16 bazu dodat je klasifikacioni deo prilagođen binarnoj klasifikaciji:

```text
Ulazna slika 224 × 224 × 3
↓
Data Augmentation
↓
VGG-16 preprocessing
↓
VGG-16 konvoluciona baza
↓
GlobalAveragePooling2D
↓
Dense(256, ReLU)
↓
Dropout(0.5)
↓
Dense(1, Sigmoid)
```

Tokom treninga korišćeni su:

- data augmentation,
- Adam optimizer,
- Binary Crossentropy,
- EarlyStopping,
- ReduceLROnPlateau.

### Rezultati Transfer Learning modela

| Metrika | Rezultat |
|---|---:|
| Accuracy | **98.10%** |
| Precision | **98.39%** |
| Recall | **97.80%** |
| F1-score | **98.09%** |

Matrica konfuzije:

```text
                 Predicted cat    Predicted dog
Actual cat             492               8
Actual dog              11             489
```

Od ukupno 1.000 test slika pravilno je klasifikovana **981 slika**.

Model se čuva kao:

```text
vgg16_transfer_learning.keras
```

---

## 04 — Fine-tuning VGG-16 modela

U notebook-u `04_finetuning.ipynb` nastavlja se treniranje prethodno sačuvanog transfer learning modela.

Umesto treniranja cele VGG-16 mreže, odmrznut je samo poslednji konvolucioni blok:

```text
block5
```

Konvolucioni slojevi koji se dodatno treniraju su:

```text
block5_conv1
block5_conv2
block5_conv3
```

Niži VGG-16 slojevi ostavljeni su zamrznuti jer uče opštije vizuelne karakteristike, kao što su ivice, teksture i jednostavni oblici.

Fine-tuning se izvršava sa znatno manjom stopom učenja:

```text
learning_rate = 1e-5
```

Tokom treninga ponovo su korišćeni:

- `EarlyStopping`,
- `ReduceLROnPlateau`.

Najbolji validation loss ostvaren je u drugoj epohi.

Nakon toga training accuracy je nastavio da raste dok se validation loss pogoršavao, što je ukazivalo na početak overfitting-a.

`EarlyStopping` je zato vratio težine iz najbolje epohe.

### Rezultati Fine-tuning modela

| Metrika | Rezultat |
|---|---:|
| Accuracy | **99.00%** |
| Precision | **99.00%** |
| Recall | **99.00%** |
| F1-score | **99.00%** |

Matrica konfuzije:

```text
                 Predicted cat    Predicted dog
Actual cat             495               5
Actual dog               5             495
```

Od ukupno 1.000 test slika pravilno je klasifikovano **990 slika**.

Fine-tuned model čuva se kao:

```text
vgg16_finetuned.keras
```

---

## Poređenje modela

Sva tri pristupa evaluirana su na istom test skupu.

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Baseline CNN | 90.30% | 89.74% | 91.00% | 90.37% |
| VGG-16 Transfer Learning | 98.10% | 98.39% | 97.80% | 98.09% |
| **VGG-16 Fine-tuning** | **99.00%** | **99.00%** | **99.00%** | **99.00%** |

Rezultati pokazuju jasno poboljšanje kroz tri faze:

```text
Baseline CNN
90.30%
    ↓
Transfer Learning VGG-16
98.10%
    ↓
Fine-tuning VGG-16
99.00%
```

Transfer learning povećao je test accuracy za **7.80 procentnih poena** u odnosu na baseline model.

Fine-tuning je zatim doneo dodatno poboljšanje od **0.90 procentnih poena**.

Ukupno poboljšanje finalnog modela u odnosu na baseline iznosi **8.70 procentnih poena**.

Grafičko poređenje Accuracy, Precision, Recall i F1-score vrednosti prikazano je u notebook-u `06_demo.ipynb`.

---

## 05 — Vizualizacija međuslojnih aktivacija

Notebook `05_visualizations_intermediate_activations.ipynb` koristi se za analizu reprezentacija koje neuronska mreža uči u različitim slojevima.

Vizualizacijom aktivacija moguće je posmatrati kako se ulazna slika transformiše prolaskom kroz mrežu.

Niži slojevi obično reaguju na jednostavnije karakteristike slike, kao što su:

- ivice,
- kontrasti,
- osnovne teksture.

Dublji slojevi predstavljaju složenije i apstraktnije karakteristike relevantne za razlikovanje klasa.

Ovakva analiza pruža bolji uvid u način na koji konvoluciona neuronska mreža gradi reprezentaciju ulazne slike.

---

## 05 — Grad-CAM vizualizacija

Notebook `05_vizualizations_grad_cam.ipynb` koristi se za interpretaciju odluka finalnog fine-tuned VGG-16 modela.

Korišćena je tehnika:

**Grad-CAM — Gradient-weighted Class Activation Mapping**

Grad-CAM omogućava vizuelizaciju oblasti ulazne slike koje su najviše doprinele konkretnoj odluci neuronske mreže.

Za Grad-CAM je korišćen poslednji konvolucioni sloj VGG-16 mreže:

```text
block5_conv3
```

Izlaz ovog sloja ima dimenzije:

```text
14 × 14 × 512
```

Grad-CAM koristi gradijente predikcije u odnosu na feature mape kako bi procenio značaj pojedinačnih oblasti slike.

Finalni model koristi sigmoid aktivaciju za binarnu klasifikaciju.

Kod veoma sigurnih predikcija sigmoid funkcija može biti zasićena, zbog čega gradijenti postaju veoma mali.

Zbog toga se Grad-CAM računa nad **logit vrednošću pre sigmoid aktivacije**, što omogućava stabilnije gradijente i kod veoma sigurnih predikcija.

Grad-CAM analiza pokazuje da se model u analiziranim primerima uglavnom fokusira na semantički relevantne oblasti životinje, kao što su:

- oči,
- njuška,
- glava,
- karakteristični delovi tela.

Rezultati ukazuju da model svoje odluke pretežno zasniva na vizuelnim karakteristikama same životinje, a ne samo na pozadini slike.

---

## 06 — Demo

Notebook `06_demo.ipynb` predstavlja završnu demonstraciju projekta.

Na početku notebook-a prikazuje se poređenje tri modela:

- Baseline CNN,
- Transfer Learning VGG-16,
- Fine-tuning VGG-16.

Poređenje obuhvata:

- Accuracy,
- Precision,
- Recall,
- F1-score.

Rezultati su prikazani tabelarno i grafički.

Nakon toga učitava se najbolji model:

```text
vgg16_finetuned.keras
```

Demo prikazuje kompletan tok:

1. izbor slike,
2. učitavanje slike,
3. promenu dimenzija slike na `224 × 224`,
4. pripremu slike za VGG-16,
5. klasifikaciju na klasu `cat` ili `dog`,
6. prikaz pouzdanosti predikcije,
7. Grad-CAM interpretaciju rezultata.

Na ovaj način završni notebook demonstrira kompletan tok od ulazne slike do klasifikacije i interpretacije odluke modela.

---

## Korišćene tehnologije

Projekat je implementiran u Python-u i razvijan prvenstveno u **Google Colab** okruženju.

Glavne korišćene biblioteke su:

- TensorFlow
- Keras
- NumPy
- Pandas
- Matplotlib
- scikit-learn

Za VGG-16 koristi se:

```python
tensorflow.keras.applications.VGG16
```

Za evaluaciju modela koriste se funkcije iz biblioteke:

```python
sklearn.metrics
```

---

## Pokretanje projekta

### 1. Kloniranje repozitorijuma

```bash
git clone https://github.com/minaristic1/Household-Animals-Classification.git
```

Zatim:

```bash
cd Household-Animals-Classification
```

### 2. Preuzimanje dataseta

Dogs vs. Cats dataset može se preuzeti sa:

https://www.kaggle.com/c/dogs-vs-cats

Tokom razvoja projekta dataset je u Google Drive-u bio organizovan na putanji:

```text
/content/drive/MyDrive/MU-projekat/data/dogs-vs-cats/train/train
```

Primer strukture:

```text
MU-projekat/
│
├── data/
│   └── dogs-vs-cats/
│       └── train/
│           └── train/
│               ├── cat.0.jpg
│               ├── cat.1.jpg
│               ├── ...
│               ├── dog.0.jpg
│               ├── dog.1.jpg
│               └── ...
│
└── models/
```

Ako se dataset nalazi na drugoj lokaciji, potrebno je promeniti `DATASET_DIR` u odgovarajućem notebook-u.

---

## Potrebne biblioteke

Za lokalno pokretanje mogu se instalirati potrebne biblioteke:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

Google Colab već sadrži većinu potrebnih biblioteka.

Za treniranje VGG-16 modela preporučuje se GPU runtime.

---

## Redosled pokretanja notebook-a

Notebook-e treba pokretati sledećim redosledom:

```text
01_data_exploration.ipynb
        ↓
02_baseline_cnn.ipynb
        ↓
03_transfer_learning.ipynb
        ↓
04_finetuning.ipynb
        ↓
05_visualizations_intermediate_activations.ipynb
        ↓
05_vizualizations_grad_cam.ipynb
        ↓
06_demo.ipynb
```

Oba `05` notebook-a pripadaju fazi interpretacije modela i nezavisno analiziraju različite aspekte ponašanja neuronske mreže.

`split.csv` mora ostati nepromenjen kako bi svi modeli koristili istu train/validation/test podelu.

---

## Sačuvani modeli

Tokom projekta kreirani su sledeći modeli:

```text
vgg6_baseline_regularized.keras
vgg16_transfer_learning.keras
vgg16_finetuned.keras
```

Modeli se tokom rada čuvaju u Google Drive direktorijumu:

```text
/content/drive/MyDrive/MU-projekat/models
```

Finalni model projekta je:

```text
vgg16_finetuned.keras
```

jer ostvaruje najbolje rezultate na test skupu.

Zbog veličine `.keras` modeli nisu uključeni direktno u GitHub repozitorijum.

---

## Reproduktivnost

Za eksperimente se koristi:

```python
SEED = 42
```

Ista podela iz `split.csv` koristi se za sve modele.

Ulazne slike za VGG-16 skaliraju se na:

```text
224 × 224 × 3
```

Batch size tokom treniranja je:

```text
32
```

Time se omogućava konzistentno poređenje različitih pristupa.

---

## Evaluacija

Za evaluaciju modela korišćene su sledeće metrike:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Kod binarne klasifikacije koristi se mapiranje:

```text
cat = 0
dog = 1
```

Prag za sigmoid izlaz je:

```text
0.5
```

---

## Podela rada

### Alma Hodžić — 1120/2025

- analiza i priprema skupa podataka,
- formiranje zajedničke train/validation/test podele,
- kreiranje `split.csv`,
- implementacija baseline CNN modela,
- analiza overfitting-a,
- implementacija regularizovane baseline verzije,
- evaluacija baseline modela,
- vizualizacija međuslojnih aktivacija,
- osnovni deo finalnog demo notebook-a,
- poređenje rezultata modela.

### Mina Ristić — 1119/2025

- implementacija VGG-16 transfer learning pristupa,
- evaluacija transfer learning modela,
- implementacija fine-tuning pristupa,
- dodatno treniranje poslednjeg VGG-16 bloka,
- evaluacija fine-tuned modela,
- poređenje transfer learning i fine-tuning pristupa,
- implementacija Grad-CAM vizualizacije,
- analiza Grad-CAM rezultata,
- Grad-CAM deo finalnog demo notebook-a.

README, završna analiza projekta i prezentacija predstavljaju zajednički deo rada.

---

## Literatura

1. Lin, *Household Animals Classification Using Deep Learning*, Stanford CS230, Winter 2020.  
   https://cs230.stanford.edu/projects_winter_2020/reports/32639344.pdf

2. Karen Simonyan, Andrew Zisserman, *Very Deep Convolutional Networks for Large-Scale Image Recognition*, 2014.  
   https://arxiv.org/abs/1409.1556

3. Ramprasaath R. Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization*, 2017.  
   https://arxiv.org/abs/1610.02391

4. TensorFlow / Keras dokumentacija  
   https://www.tensorflow.org/

5. Keras VGG-16 dokumentacija  
   https://keras.io/api/applications/vgg/

6. Dogs vs. Cats dataset — Kaggle  
   https://www.kaggle.com/c/dogs-vs-cats

---

## Zaključak

U projektu je analiziran problem binarne klasifikacije slika pasa i mačaka kroz tri različita pristupa dubokog učenja.

Baseline CNN model treniran od početka ostvario je test accuracy od **90.30%**.

Korišćenjem transfer learning-a sa VGG-16 mrežom prethodno treniranom na ImageNet skupu rezultat je povećan na **98.10%**.

Dodatnim fine-tuning-om poslednjeg VGG-16 konvolucionog bloka ostvaren je najbolji rezultat projekta sa test accuracy od **99.00%**, uz Precision, Recall i F1-score od **99.00%**.

Vizualizacija međuslojnih aktivacija omogućava uvid u način na koji mreža postepeno gradi složenije reprezentacije ulazne slike.

Grad-CAM analiza dodatno omogućava interpretaciju konačnih odluka modela i pokazuje da se model u analiziranim primerima uglavnom fokusira na relevantne delove životinje, kao što su lice, oči i njuška.

Rezultati projekta pokazuju značaj transfer learning-a i fine-tuning-a kod problema klasifikacije slika, kao i značaj tehnika interpretabilnosti za razumevanje ponašanja dubokih neuronskih mreža.
