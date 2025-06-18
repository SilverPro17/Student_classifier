from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import io
import base64
import os
from werkzeug.utils import secure_filename
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configurações
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file siz
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Criar pasta de uploads se não existir
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

class ImageClassifier:
    def __init__(self, model_path='best_student_classifier.h5'):
        self.model_path = model_path
        self.model = None
        self.img_size = (224, 224)
        self.class_names = ['aluno_target', 'outras_pessoas']
        self.load_model()
    
    def load_model(self):
        """Carregar modelo treinado"""
        try:
            if os.path.exists(self.model_path):
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Modelo carregado: {self.model_path}")
            else:
                # Tentar carregar modelo alternativo
                alt_path = 'student_classifier_final.h5'
                if os.path.exists(alt_path):
                    self.model = tf.keras.models.load_model(alt_path)
                    logger.info(f"Modelo alternativo carregado: {alt_path}")
                else:
                    logger.error("Nenhum modelo encontrado!")
                    self.model = None
        except Exception as e:
            logger.error(f"Erro ao carregar modelo: {e}")
            self.model = None
    
    def preprocess_image(self, image):
        """Pré-processar imagem para predição"""
        try:
            # Converter para RGB se necessário
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Já está em RGB
                img = image
            elif len(image.shape) == 3 and image.shape[2] == 4:
                # RGBA para RGB
                img = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                # Grayscale para RGB
                img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            
            # Redimensionar mantendo aspect ratio
            h, w = img.shape[:2]
            if h > w:
                new_h, new_w = self.img_size[0], int(w * self.img_size[0] / h)
            else:
                new_h, new_w = int(h * self.img_size[1] / w), self.img_size[1]
            
            img = cv2.resize(img, (new_w, new_h))
            
            # Padding para completar tamanho
            delta_w = self.img_size[1] - new_w
            delta_h = self.img_size[0] - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)
            
            img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])
            
            # Normalizar pixels (0-1)
            img = img.astype(np.float32) / 255.0
            
            # Adicionar dimensão batch
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"Erro no pré-processamento: {e}")
            return None
    
    def predict(self, image):
        """Fazer predição na imagem"""
        if self.model is None:
            return {
                'error': 'Modelo não carregado',
                'success': False
            }
        
        try:
            # Pré-processar imagem
            processed_img = self.preprocess_image(image)
            if processed_img is None:
                return {
                    'error': 'Erro no pré-processamento da imagem',
                    'success': False
                }
            
            # Fazer predição
            predictions = self.model.predict(processed_img, verbose=0)
            
            # Processar resultados
            confidence_scores = predictions[0]
            predicted_class_idx = np.argmax(confidence_scores)
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(confidence_scores[predicted_class_idx])
            
            # Resultado detalhado
            result = {
                'success': True,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_target_student': predicted_class == 'aluno_target',
                'confidence_percentage': f"{confidence * 100:.1f}%",
                'all_probabilities': {
                    self.class_names[i]: float(confidence_scores[i])
                    for i in range(len(self.class_names))
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na predição: {e}")
            return {
                'error': f'Erro na predição: {str(e)}',
                'success': False
            }

# Inicializar classificador
classifier = ImageClassifier()

def allowed_file(filename):
    """Verificar se arquivo é permitido"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/health')
def health_check():
    """Verificar saúde da API"""
    model_loaded = classifier.model is not None
    return jsonify({
        'status': 'healthy' if model_loaded else 'unhealthy',
        'model_loaded': model_loaded,
        'message': 'API funcionando' if model_loaded else 'Modelo não carregado'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint para predição"""
    try:
        # Verificar se arquivo foi enviado
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo enviado'
            }), 400
        
        file = request.files['file']
        
        # Verificar se arquivo foi selecionado
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'Nenhum arquivo selecionado'
            }), 400
        
        # Verificar extensão do arquivo
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'error': 'Tipo de arquivo não permitido. Use: PNG, JPG, JPEG, GIF'
            }), 400
        
        # Ler e processar imagem
        image_bytes = file.read()
        
        # Converter bytes para imagem OpenCV
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Não foi possível processar a imagem'
            }), 400
        
        # Converter BGR para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Fazer predição
        result = classifier.predict(image)
        
        # Adicionar informações extras
        if result['success']:
            # Interpretar resultado
            if result['is_target_student']:
                if result['confidence'] > 0.8:
                    interpretation = "Alta confiança: É o aluno específico"
                elif result['confidence'] > 0.6:
                    interpretation = "Média confiança: Provavelmente é o aluno específico"
                else:
                    interpretation = "Baixa confiança: Pode ser o aluno específico"
            else:
                if result['confidence'] > 0.8:
                    interpretation = "Alta confiança: NÃO é o aluno específico"
                elif result['confidence'] > 0.6:
                    interpretation = "Média confiança: Provavelmente NÃO é o aluno específico"
                else:
                    interpretation = "Baixa confiança: Pode não ser o aluno específico"
            
            result['interpretation'] = interpretation
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro no endpoint predict: {e}")
        return jsonify({
            'success': False,
            'error': f'Erro interno do servidor: {str(e)}'
        }), 500

@app.route('/predict_base64', methods=['POST'])
def predict_base64():
    """Endpoint para predição com imagem em base64"""
    try:
        data = request.get_json()
        
        if 'image' not in data:
            return jsonify({
                'success': False,
                'error': 'Campo "image" não encontrado no JSON'
            }), 400
        
        # Decodificar base64
        image_data = data['image']
        
        # Remover prefixo data:image se presente
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decodificar
        image_bytes = base64.b64decode(image_data)
        
        # Converter para imagem
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False,
                'error': 'Não foi possível processar a imagem base64'
            }), 400
        
        # Converter BGR para RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Fazer predição
        result = classifier.predict(image)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Erro no endpoint predict_base64: {e}")
        return jsonify({
            'success': False,
            'error': f'Erro interno do servidor: {str(e)}'
        }), 500

if __name__ == '__main__':
    print("=== BACKEND FLASK - CLASSIFICADOR DE ESTUDANTE ===")
    print("Servidor iniciando...")
    
    # Verificar se modelo existe
    if classifier.model is None:
        print("AVISO: Nenhum modelo encontrado!")
        print("Certifique-se de ter treinado o modelo primeiro.")
    else:
        print("Modelo carregado com sucesso!")
    
    print("Acesse: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)