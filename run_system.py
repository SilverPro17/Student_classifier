#!/usr/bin/env python3
"""
Sistema Completo de Classificação de Estudante
Desenvolvido com TensorFlow, Flask e OpenCV

Este script oferece um menu interativo para:
1. Criar dataset com data augmentation
2. Treinar modelo CNN
3. Executar servidor Flask
4. Executar sistema completo
"""

import os
import sys
import subprocess
import argparse

def print_banner():
    """Exibir banner do sistema"""
    print("="*60)
    print("🎓 SISTEMA DE CLASSIFICAÇÃO DE ESTUDANTE")
    print("="*60)
    print("Desenvolvido com TensorFlow, Flask e OpenCV")
    print("Sistema completo para identificação de aluno específico")
    print("="*60)

def check_requirements():
    """Verificar se as dependências estão instaladas"""
    try:
        import tensorflow
        import flask
        import cv2
        import numpy
        import sklearn
        print("✅ Todas as dependências estão instaladas!")
        return True
    except ImportError as e:
        print(f"❌ Dependência não encontrada: {e}")
        print("Execute: pip install -r requirements.txt")
        return False

def create_folder_structure():
    """Criar estrutura de pastas necessárias"""
    folders = [
        "dataset/raw/aluno_target",
        "dataset/raw/outras_pessoas",
        "uploads",
        "templates",
        "logs"
    ]
    
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    print("✅ Estrutura de pastas criada!")

def create_dataset():
    """Executar criação do dataset"""
    print("\n🔄 Criando dataset com data augmentation...")
    
    # Verificar se existem imagens
    aluno_path = "dataset/raw/aluno_target"
    outras_path = "dataset/raw/outras_pessoas"
    
    if not os.path.exists(aluno_path) or not os.path.exists(outras_path):
        print("❌ Pastas de dataset não encontradas!")
        print(f"Crie as pastas: {aluno_path} e {outras_path}")
        return False
    
    aluno_imgs = [f for f in os.listdir(aluno_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    outras_imgs = [f for f in os.listdir(outras_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(aluno_imgs) == 0 or len(outras_imgs) == 0:
        print("❌ Adicione imagens nas pastas antes de continuar!")
        print(f"- {aluno_path}: {len(aluno_imgs)} imagens")
        print(f"- {outras_path}: {len(outras_imgs)} imagens")
        return False
    
    print(f"📊 Imagens encontradas:")
    print(f"- Aluno específico: {len(aluno_imgs)} imagens")
    print(f"- Outras pessoas: {len(outras_imgs)} imagens")
    
    # Executar script de criação do dataset
    try:
        from dataset_creation import DatasetCreator
        creator = DatasetCreator()
        creator.create_dataset()
        return True
    except Exception as e:
        print(f"❌ Erro ao criar dataset: {e}")
        return False

def train_model():
    """Executar treinamento do modelo"""
    print("\n🔄 Treinando modelo de classificação...")
    
    # Verificar se dataset processado existe
    if not os.path.exists("dataset/processed/train"):
        print("❌ Dataset processado não encontrado!")
        print("Execute primeiro a opção 1 (Criar Dataset)")
        return False
    
    try:
        from model_training import StudentClassifier
        classifier = StudentClassifier()
        classifier.train_model(epochs=30)
        classifier.plot_training_history()
        classifier.evaluate_model()
        return True
    except Exception as e:
        print(f"❌ Erro no treinamento: {e}")
        return False

def run_flask_server():
    """Executar servidor Flask"""
    print("\n🔄 Iniciando servidor Flask...")
    
    # Verificar se modelo existe
    model_files = ['best_student_classifier.h5', 'student_classifier_final.h5']
    model_exists = any(os.path.exists(f) for f in model_files)
    
    if not model_exists:
        print("❌ Modelo não encontrado!")
        print("Execute primeiro a opção 2 (Treinar Modelo)")
        return False
    
    # Criar template HTML se não existir
    if not os.path.exists("templates/index.html"):
        print("📝 Criando template HTML...")
        # Aqui você colocaria o código para criar o arquivo HTML
        # Por simplicidade, assumimos que já existe
    
    try:
        # Executar servidor Flask
        print("🚀 Servidor Flask iniciando em http://localhost:5000")
        print("Pressione Ctrl+C para parar o servidor")
        
        from flask_backend import app
        app.run(debug=True, host='0.0.0.0', port=5000)
        
    except KeyboardInterrupt:
        print("\n✅ Servidor parado pelo usuário")
    except Exception as e:
        print(f"❌ Erro no servidor: {e}")
        return False

def run_complete_system():
    """Executar sistema completo"""
    print("\n🔄 Executando sistema completo...")
    
    # 1. Verificar e criar dataset se necessário
    if not os.path.exists("dataset/processed/train"):
        print("1️⃣ Criando dataset...")
        if not create_dataset():
            return False
    else:
        print("1️⃣ ✅ Dataset já existe")
    
    # 2. Verificar e treinar modelo se necessário
    model_files = ['best_student_classifier.h5', 'student_classifier_final.h5']
    model_exists = any(os.path.exists(f) for f in model_files)
    
    if not model_exists:
        print("2️⃣ Treinando modelo...")
        if not train_model():
            return False
    else:
        print("2️⃣ ✅ Modelo já existe")
    
    # 3. Executar servidor
    print("3️⃣ Iniciando servidor...")
    run_flask_server()

def show_menu():
    """Exibir menu principal"""
    print("\n📋 MENU PRINCIPAL")
    print("1. 📊 Criar Dataset (com data augmentation)")
    print("2. 🧠 Treinar Modelo CNN")
    print("3. 🌐 Executar Servidor Flask")
    print("4. 🚀 Executar Sistema Completo")
    print("5. ❓ Ajuda e Instruções")
    print("6. 🚪 Sair")
    print("-" * 40)

def show_help():
    """Exibir instruções de uso"""
    print("\n📖 INSTRUÇÕES DE USO")
    print("="*50)
    
    print("\n1️⃣ PREPARAÇÃO DO DATASET:")
    print("• Crie as pastas: dataset/raw/aluno_target/ e dataset/raw/outras_pessoas/")
    print("• Adicione pelo menos 10-20 fotos do aluno específico em 'aluno_target'")
    print("• Adicione pelo menos 50-100 fotos de outras pessoas em 'outras_pessoas'")
    print("• Use formatos: PNG, JPG, JPEG")
    print("• Resolução recomendada: 224x224 ou maior")
    
    print("\n2️⃣ TREINAMENTO:")
    print("• O sistema usará transfer learning com MobileNetV2")
    print("• Aplicará data augmentation automaticamente")
    print("• Salvará o melhor modelo durante o treinamento")
    print("• Tempo estimado: 10-30 minutos (dependendo do hardware)")
    
    print("\n3️⃣ USO DO SISTEMA:")
    print("• Acesse http://localhost:5000 no navegador")
    print("• Faça upload de uma imagem")
    print("• O sistema retorna se é o aluno específico ou não")
    print("• Inclui percentual de confiança na predição")
    
    print("\n🔧 REQUISITOS:")
    print("• Python 3.7+")
    print("• TensorFlow 2.x")
    print("• OpenCV")
    print("• Flask")
    print("• Execute: pip install -r requirements.txt")

def main():
    """Função principal"""
    print_banner()
    
    # Verificar dependências
    if not check_requirements():
        return
    
    # Criar estrutura de pastas
    create_folder_structure()
    
    # Menu principal
    while True:
        show_menu()
        choice = input("Escolha uma opção (1-6): ").strip()
        
        if choice == '1':
            create_dataset()
        elif choice == '2':
            train_model()
        elif choice == '3':
            run_flask_server()
        elif choice == '4':
            run_complete_system()
        elif choice == '5':
            show_help()
        elif choice == '6':
            print("👋 Saindo do sistema...")
            break
        else:
            print("❌ Opção inválida! Escolha entre 1-6.")
        
        input("\nPressione Enter para continuar...")

if __name__ == "__main__":
    # Suporte para argumentos de linha de comando
    parser = argparse.ArgumentParser(description='Sistema de Classificação de Estudante')
    parser.add_argument('--mode', choices=['dataset', 'train', 'server', 'complete'], 
                       help='Modo de execução direto')
    
    args = parser.parse_args()
    
    if args.mode:
        print_banner()
        if not check_requirements():
            sys.exit(1)
        create_folder_structure()
        
        if args.mode == 'dataset':
            create_dataset()
        elif args.mode == 'train':
            train_model()
        elif args.mode == 'server':
            run_flask_server()
        elif args.mode == 'complete':
            run_complete_system()
    else:
        main()