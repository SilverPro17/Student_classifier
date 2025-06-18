import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
import hashlib
import re

class DatasetCreator:
    def __init__(self, base_path="dataset"):
        self.base_path = base_path
        self.img_size = (224, 224)
        
        # Criar estrutura de pastas
        self.create_folder_structure()
        
    def create_folder_structure(self):
        """Criar estrutura de pastas para o dataset"""
        folders = [
            f"{self.base_path}/raw/aluno_target",
            f"{self.base_path}/raw/outras_pessoas", 
            f"{self.base_path}/processed/train/aluno_target",
            f"{self.base_path}/processed/train/outras_pessoas",
            f"{self.base_path}/processed/validation/aluno_target",
            f"{self.base_path}/processed/validation/outras_pessoas",
            f"{self.base_path}/processed/test/aluno_target",
            f"{self.base_path}/processed/test/outras_pessoas"
        ]
        
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
            
    def sanitize_filename(self, filename):
        """Limpar nome do arquivo para evitar problemas no Windows"""
        # Remover caracteres especiais e limitar tamanho
        name, ext = os.path.splitext(filename)
        # Remover caracteres problemáticos
        name = re.sub(r'[<>:"/\\|?*]', '_', name)
        # Limitar tamanho do nome
        if len(name) > 50:
            # Usar hash para nomes muito longos
            name_hash = hashlib.md5(name.encode()).hexdigest()[:8]
            name = f"img_{name_hash}"
        return f"{name}{ext}"
    
    def preprocess_image(self, image_path, output_size=(224, 224)):
        """Pré-processar uma imagem individual"""
        try:
            # Carregar imagem
            img = cv2.imread(image_path)
            if img is None:
                print(f"AVISO: Não foi possível carreg {image_path}")
                return None
                
            # Converter BGR para RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Redimensionar mantendo aspect ratio
            h, w = img.shape[:2]
            if h > w:
                new_h, new_w = output_size[0], int(w * output_size[0] / h)
            else:
                new_h, new_w = int(h * output_size[1] / w), output_size[1]
                
            img = cv2.resize(img, (new_w, new_h))
            
            # Padding para completar o tamanho desejado
            delta_w = output_size[1] - new_w
            delta_h = output_size[0] - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            img = cv2.copyMakeBorder(img, top, bottom, left, right, 
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            return img
            
        except Exception as e:
            print(f"Erro ao processar {image_path}: {e}")
            return None
    
    def apply_data_augmentation(self, input_folder, output_folder, num_augmented=5):
        """Aplicar data augmentation com nomes de arquivo seguros"""
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            brightness_range=[0.8, 1.2],
            fill_mode='nearest'
        )
        
        print(f"Processando imagens de: {input_folder}")
        
        # Verificar se pasta existe
        if not os.path.exists(input_folder):
            print(f"ERRO: Pasta não encontrada: {input_folder}")
            return
        
        # Processar cada imagem na pasta
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Encontradas {len(image_files)} imagens")
        
        for idx, filename in enumerate(image_files):
            print(f"Processando {idx+1}/{len(image_files)}: {filename}")
            
            img_path = os.path.join(input_folder, filename)
            
            # Pré-processar imagem original
            img = self.preprocess_image(img_path)
            if img is None:
                continue
            
            # Nome seguro para o arquivo
            safe_name = self.sanitize_filename(filename)
            base_name = os.path.splitext(safe_name)[0]
            
            # Salvar imagem original processada
            original_name = f"orig_{idx:03d}_{base_name}.jpg"
            original_path = os.path.join(output_folder, original_name)
            cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            
            # Gerar versões aumentadas
            img_array = img_to_array(img)
            img_array = img_array.reshape((1,) + img_array.shape)
            
            aug_count = 0
            for batch in datagen.flow(img_array, batch_size=1):
                if aug_count >= num_augmented:
                    break
                
                # Nome seguro para arquivo aumentado
                aug_name = f"aug_{idx:03d}_{aug_count:02d}_{base_name}.jpg"
                aug_path = os.path.join(output_folder, aug_name)
                
                # Salvar imagem aumentada
                aug_img = batch[0].astype(np.uint8)
                cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))
                
                aug_count += 1
        
        print(f"Processamento concluído para {input_folder}")
                        
    def create_dataset(self):
        """Criar dataset completo com train/validation/test split"""
        print("3 Iniciando criação do dataset...")
        
        # Verificar se existem imagens
        aluno_path = f"{self.base_path}/raw/aluno_target"
        outras_path = f"{self.base_path}/raw/outras_pessoas"
        
        # Contar imagens
        aluno_imgs = []
        outras_imgs = []
        
        if os.path.exists(aluno_path):
            aluno_imgs = [f for f in os.listdir(aluno_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if os.path.exists(outras_path):
            outras_imgs = [f for f in os.listdir(outras_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f" Imagens encontradas:")
        print(f"- Aluno específico: {len(aluno_imgs)} imagens")
        print(f"- Outras pessoas: {len(outras_imgs)} imagens")
        
        if len(aluno_imgs) == 0 or len(outras_imgs) == 0:
            print("❌ Erro: Adicione imagens nas pastas antes de continuar!")
            return False
        
        # Criar pastas temporárias
        temp_aluno = f"{self.base_path}/temp/aluno_target"
        temp_outras = f"{self.base_path}/temp/outras_pessoas"
        os.makedirs(temp_aluno, exist_ok=True)
        os.makedirs(temp_outras, exist_ok=True)
        
        print(" Aplicando data augmentation...")
        
        # Aplicar augmentation para cada classe
        self.apply_data_augmentation(aluno_path, temp_aluno, num_augmented=5)
        self.apply_data_augmentation(outras_path, temp_outras, num_augmented=3)
        
        print(" Dividindo dataset em train/validation/test...")
        
        # Dividir em train/val/test para cada classe
        for class_name in ['aluno_target', 'outras_pessoas']:
            temp_folder = f"{self.base_path}/temp/{class_name}"
            
            if not os.path.exists(temp_folder):
                print(f"AVISO: Pasta temporária não encontrada: {temp_folder}")
                continue
                
            images = [f for f in os.listdir(temp_folder) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            print(f"Classe {class_name}: {len(images)} imagens processadas")
            
            if len(images) == 0:
                print(f"AVISO: Nenhuma imagem encontrada para {class_name}")
                continue
            
            # Split 70% train, 20% validation, 10% test
            if len(images) >= 10:
                train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
                val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.33, random_state=42)
            else:
                # Para datasets muito pequenos
                train_imgs = images[:max(1, int(len(images) * 0.7))]
                val_imgs = images[len(train_imgs):max(len(train_imgs)+1, int(len(images) * 0.9))]
                test_imgs = images[len(train_imgs)+len(val_imgs):]
                if len(test_imgs) == 0:
                    test_imgs = [images[-1]]  # Pelo menos 1 para teste
            
            # Mover arquivos para pastas corretas
            splits = {
                'train': train_imgs,
                'validation': val_imgs, 
                'test': test_imgs
            }
            
            for split_name, img_list in splits.items():
                dest_folder = f"{self.base_path}/processed/{split_name}/{class_name}"
                print(f"Movendo {len(img_list)} imagens para {split_name}/{class_name}")
                
                for img_name in img_list:
                    src = os.path.join(temp_folder, img_name)
                    dst = os.path.join(dest_folder, img_name)
                    
                    try:
                        shutil.copy2(src, dst)
                    except Exception as e:
                        print(f"Erro ao copiar {img_name}: {e}")
        
        # Limpar pasta temporária
        try:
            if os.path.exists(f"{self.base_path}/temp"):
                shutil.rmtree(f"{self.base_path}/temp")
        except Exception as e:
            print(f"Aviso: Não foi possível remover pasta temp: {e}")
            
        print(" Dataset criado com sucesso!")
        self.print_dataset_stats()
        return True
        
    def print_dataset_stats(self):
        """Imprimir estatísticas do dataset"""
        print("\n=== ESTATÍSTICAS DO DATASET ===")
        
        total_train = 0
        total_val = 0
        total_test = 0
        
        for split in ['train', 'validation', 'test']:
            print(f"\n{split.upper()}:")
            split_total = 0
            for class_name in ['aluno_target', 'outras_pessoas']:
                folder = f"{self.base_path}/processed/{split}/{class_name}"
                if os.path.exists(folder):
                    count = len([f for f in os.listdir(folder) 
                               if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    print(f"  {class_name}: {count} imagens")
                    split_total += count
                else:
                    print(f"  {class_name}: 0 imagens (pasta não encontrada)")
            
            print(f"  TOTAL {split}: {split_total} imagens")
            
            if split == 'train':
                total_train = split_total
            elif split == 'validation':
                total_val = split_total
            elif split == 'test':
                total_test = split_total
        
        print(f"\n RESUMO GERAL:")
        print(f"Total de imagens: {total_train + total_val + total_test}")
        print(f"Train: {total_train} | Validation: {total_val} | Test: {total_test}")

# Exemplo de uso
if __name__ == "__main__":
    # Criar o dataset
    creator = DatasetCreator()
    
    print("INSTRUÇÕES PARA USO:")
    print("1. Coloque as fotos do aluno específico na pasta: dataset/raw/aluno_target/")
    print("2. Coloque fotos de outras pessoas na pasta: dataset/raw/outras_pessoas/")
    print("3. Execute este script para processar e criar o dataset")
    print("\n Processando dataset...")
    
    # Verificar se existem imagens nas pastas raw
    aluno_path = "dataset/raw/aluno_target"
    outras_path = "dataset/raw/outras_pessoas"
    
    if os.path.exists(aluno_path) and os.path.exists(outras_path):
        aluno_count = len([f for f in os.listdir(aluno_path) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        outras_count = len([f for f in os.listdir(outras_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        
        if aluno_count > 0 and outras_count > 0:
            success = creator.create_dataset()
            if success:
                print("\n Dataset criado com sucesso!")
            else:
                print("\n Erro na criação do dataset!")
        else:
            print("AVISO: Adicione imagens nas pastas antes de executar!")
            print(f"Aluno específico: {aluno_count} imagens")
            print(f"Outras pessoas: {outras_count} imagens")
    else:
        print("AVISO: Crie as pastas e adicione imagens antes de executar!")
        print(f"Verificar: {aluno_path}")
        print(f"Verificar: {outras_path}")