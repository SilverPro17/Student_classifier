import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

class StudentClassifier:
    def __init__(self, dataset_path="dataset/processed", img_size=(224, 224)):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = 16  # Reduzido para evitar problemas de memória
        self.num_classes = 2
        self.model = None
        self.history = None
        
    def check_dataset(self):
        """Verificar se o dataset existe e tem imagens"""
        print(" Verificando dataset...")
        
        splits = ['train', 'validation', 'test']
        classes = ['aluno_target', 'outras_pessoas']
        
        total_images = 0
        dataset_ok = True
        
        for split in splits:
            split_total = 0
            print(f"\n{split.upper()}:")
            
            for class_name in classes:
                folder_path = os.path.join(self.dataset_path, split, class_name)
                
                if os.path.exists(folder_path):
                    images = [f for f in os.listdir(folder_path) 
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    count = len(images)
                    print(f"  {class_name}: {count} imagens")
                    split_total += count
                else:
                    print(f"  {class_name}: 0 imagens (pasta não encontrada)")
                    count = 0
                
                if split == 'train' and count == 0:
                    dataset_ok = False
            
            print(f"  Total {split}: {split_total}")
            total_images += split_total
        
        print(f"\n Total geral: {total_images} imagens")
        
        if not dataset_ok or total_images == 0:
            print(" Dataset inválido! Execute primeiro a criação do dataset.")
            return False
        
        if total_images < 10:
            print(" Dataset muito pequeno. Recomenda-se pelo menos 20+ imagens.")
        
        return True
        
    def create_data_generators(self):
        """Criar geradores de dados para train/validation/test"""
        
        # Verificar dataset primeiro
        if not self.check_dataset():
            return None, None, None
        
        # Gerador para treinamento com normalização e augmentation leve
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Gerador para validação e teste (apenas normalização)
        val_test_datagen = ImageDataGenerator(rescale=1./255)
        
        try:
            # Carregar dados
            train_generator = train_datagen.flow_from_directory(
                os.path.join(self.dataset_path, 'train'),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=True
            )
            
            validation_generator = val_test_datagen.flow_from_directory(
                os.path.join(self.dataset_path, 'validation'),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            test_generator = val_test_datagen.flow_from_directory(
                os.path.join(self.dataset_path, 'test'),
                target_size=self.img_size,
                batch_size=self.batch_size,
                class_mode='categorical',
                shuffle=False
            )
            
            # Verificar se os geradores têm dados
            if train_generator.samples == 0:
                print("❌ Erro: Nenhuma imagem de treino encontrada!")
                return None, None, None
            
            print(f" Geradores criados:")
            print(f"  Train: {train_generator.samples} imagens")
            print(f"  Validation: {validation_generator.samples} imagens") 
            print(f"  Test: {test_generator.samples} imagens")
            print(f"  Classes: {list(train_generator.class_indices.keys())}")
            
            return train_generator, validation_generator, test_generator
            
        except Exception as e:
            print(f" Erro ao criar geradores: {e}")
            return None, None, None
    
    def create_model(self):
        """Criar modelo CNN com transfer learning"""
        print("🏗️ Criando modelo...")
        
        # Base model com MobileNetV2 (pré-treinado)
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )
        
        # Congelar camadas do modelo base
        base_model.trainable = False
        
        # Adicionar camadas customizadas
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.1),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        print(" Modelo criado com sucesso!")
        return model
    
    def compile_model(self, model):
        """Compilar modelo com otimizador e métricas"""
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    
    def create_callbacks(self):
        """Criar callbacks para treinamento"""
        callbacks_list = [
            # Early stopping
            callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduzir learning rate
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=7,
                min_lr=1e-7,
                verbose=1
            ),
            
            # Salvar melhor modelo
            callbacks.ModelCheckpoint(
                'best_student_classifier.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        return callbacks_list
    
    def train_model(self, epochs=50):
        """Treinar o modelo"""
        print(" Iniciando treinamento do modelo...")
        
        # Carregar dados
        train_gen, val_gen, test_gen = self.create_data_generators()
        
        if train_gen is None:
            print(" Erro: Não foi possível carregar o dataset!")
            return None
        
        # Ajustar épocas baseado no tamanho do dataset
        if train_gen.samples < 50:
            epochs = min(epochs, 30)
            print(f" Dataset pequeno. Reduzindo épocas para {epochs}")
        
        # Criar modelo
        self.model = self.create_model()
        self.model = self.compile_model(self.model)
        
        print(" Resumo do modelo:")
        self.model.summary()
        
        print(f"🏃 Iniciando treinamento por {epochs} épocas...")
        
        try:
            self.history = self.model.fit(
                train_gen,
                epochs=epochs,
                validation_data=val_gen,
                callbacks=self.create_callbacks(),
                verbose=1
            )
            
            print(" Avaliando no conjunto de teste...")
            test_results = self.model.evaluate(test_gen, verbose=0)
            print(f" Acurácia no teste: {test_results[1]:.4f}")
            
            # Salvar modelo final
            self.model.save('student_classifier_final.h5')
            print(" Modelo salvo como 'student_classifier_final.h5'")
            
            return self.history
            
        except Exception as e:
            print(f" Erro durante o treinamento: {e}")
            return None
    
    def plot_training_history(self):
        """Plotar histórico de treinamento"""
        if self.history is None:
            print(" Modelo ainda não foi treinado!")
            return
            
        plt.figure(figsize=(12, 4))
        
        # Acurácia
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['accuracy'], label='Treino', linewidth=2)
        plt.plot(self.history.history['val_accuracy'], label='Validação', linewidth=2)
        plt.title('Acurácia do Modelo', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Acurácia')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Loss
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['loss'], label='Treino', linewidth=2)
        plt.plot(self.history.history['val_loss'], label='Validação', linewidth=2)
        plt.title('Loss do Modelo', fontsize=14, fontweight='bold')
        plt.xlabel('Época')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print(" Gráfico salvo como 'training_history.png'")
        plt.show()
    
    def evaluate_model(self):
        """Avaliar modelo no conjunto de teste"""
        if self.model is None:
            print(" Modelo ainda não foi treinado!")
            return
            
        print(" Avaliando modelo...")
        
        _, _, test_gen = self.create_data_generators()
        
        if test_gen is None or test_gen.samples == 0:
            print(" Conjunto de teste não disponível!")
            return
        
        try:
            # Predições
            test_gen.reset()  # Reset generator
            predictions = self.model.predict(test_gen, verbose=1)
            predicted_classes = np.argmax(predictions, axis=1)
            
            # Classes verdadeiras
            true_classes = test_gen.classes[:len(predicted_classes)]
            
            # Nomes das classes
            class_names = list(test_gen.class_indices.keys())
            
            # Relatório de classificação
            print("\n Relatório de Classificação:")
            print(classification_report(true_classes, predicted_classes, 
                                      target_names=class_names))
            
            # Matriz de confusão
            cm = confusion_matrix(true_classes, predicted_classes)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Matriz de Confusão', fontsize=14, fontweight='bold')
            plt.xlabel('Predição')
            plt.ylabel('Verdadeiro')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
            print(" Matriz de confusão salva como 'confusion_matrix.png'")
            plt.show()
            
        except Exception as e:
            print(f"❌ Erro na avaliação: {e}")

def main():
    """Função principal para treinar o modelo"""
    print("=== TREINAMENTO DO CLASSIFICADOR DE ESTUDANTE ===")
    
    # Verificar se dataset existe
    dataset_path = "dataset/processed"
    if not os.path.exists(os.path.join(dataset_path, "train")):
        print(" ERRO: Dataset não encontrado!")
        print("Execute primeiro o script de criação do dataset (opção 1 no menu)")
        print(f"Esperado: {os.path.join(dataset_path, 'train')}")
        return False
    
    # Criar e treinar classificador
    classifier = StudentClassifier(dataset_path)
    
    # Treinar modelo
    history = classifier.train_model(epochs=40)
    
    if history is not None:
        # Plotar resultados
        classifier.plot_training_history()
        classifier.evaluate_model()
        
        print("\n Treinamento concluído com sucesso!")
        print(" Arquivos gerados:")
        print("- best_student_classifier.h5 (melhor modelo)")
        print("- student_classifier_final.h5 (modelo final)")
        print("- training_history.png (gráficos)")
        print("- confusion_matrix.png (matriz de confusão)")
        return True
    else:
        print("\n❌ Treinamento falhou!")
        return False

if __name__ == "__main__":
    main()