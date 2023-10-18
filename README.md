# INTRODUCCION-A-LA-CUANTIZACION

## 1- Investigación previa: 

Introducción a la cuantización ( está en los archivos pdf del repositorio) 



# 


## 2-Configuración inicial:
Obtenga un modelo preentrenado GPT-2.
Establezca una métrica base del rendimiento del modelo sin cuantizar.
El código con su respuesta es el siguiente ( dejaré el archivo arriba (Untitled1 (1).ipynb), y el link del google colab en el que he realizado el código, en el cual para ejecutarlo, no hay que olvidarse de escribir " !pip install transformers" ) : 
```
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Cargar un modelo preentrenado de GPT-2 en español
model_name = "datificate/gpt2-small-spanish"  # GPT-2 preentrenado en español
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Texto de entrada en español
input_text = "Me gusta mucho el futbol"

# Tokenizar el texto de entrada
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generar texto adicional basado en la entrada
output = model.generate(
    input_ids,
    max_length=150,  # Ajusta la longitud del texto generado según sea necesario
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

# Decodificar el texto generado en español
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Mostrar el texto generado en español
print("Texto generado:\n", generated_text)
```


## Que tiene como respuesta a este ejemplo concreto: 
# 
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Texto generado:
 Me gusta mucho el futbol, pero no lo es todo.

En el año 2000, el club se mudó a la ciudad de Nueva York, donde se convirtió en el primer equipo de fútbol americano en jugar en la Major League Soccer. En el 2000 se trasladó a Los Ángeles, California, y en 2002 se mudaron a San Francisco, San Diego, Los Angeles y Los San Antonio. El equipo se estableció en Los Santos, Texas, en 2003, con el nombre de "Los Angeles Galaxy".
El equipo jugó en las series de la MLS, la Copa MLS y la Liga de Campeones de CONCACAF. Los equipos de los Ángeles se convirtieron en los primeros equipos en ganar la liga en su historia. La franquicia se fundó en 2004.
# 
## Explicación del código:
# 
Este código utiliza un modelo de lenguaje GPT-2 preentrenado en español para generar texto autónomamente a partir de una entrada dada. A continuación, se describen las principales acciones realizadas:

Carga del Modelo y Tokenizador: El código carga un modelo GPT-2 preentrenado específico para el español ("datificate/gpt2-small-spanish") y el tokenizador asociado. Estos componentes son esenciales para la generación de texto.

Texto de Entrada: Se proporciona un fragmento de texto en español como entrada. En este caso, el texto de entrada es "Me gusta mucho el futbol."

Tokenización: El texto de entrada se convierte en una secuencia de tokens utilizando el tokenizador. Esta representación tokenizada es lo que el modelo utilizará para generar texto coherente.

Generación de Texto: El modelo GPT-2 se utiliza para generar texto adicional basado en la entrada proporcionada. Se especifican diversas configuraciones, como la longitud máxima del texto generado, el número de secuencias de salida, y parámetros que afectan la creatividad del texto generado.

Decodificación y Visualización: El texto generado se decodifica desde la representación tokenizada a texto legible en español. Finalmente, el texto generado se imprime en la consola.

## Link del código:
https://colab.research.google.com/drive/19xZHwaZr86gOshX2SxreeqZTm_kjdv2D#scrollTo=0ct5SK8s417K

# 

## 3-Representación de Coma Flotante
Representación en coma flotante.pdf (Está en los archivos del repositorio)


## 4- Implementación PTQ: Aplique la técnica PTQ al modelo GPT-2. Evalúe el rendimiento del modelo después de la cuantización
El código para llevralo a cabo es el siguiente (esta en uno de los archivos del repositorio llamado "Untitled3.ipynb"):
```
!pip install torch
!pip install transformers
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.quantization import QuantStub, DeQuantStub, default_qconfig

# Cargar un modelo preentrenado de GPT-2 en español
model_name = "datificate/gpt2-small-spanish"  # GPT-2 preentrenado en español
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Texto de entrada en español
input_text = "Me gusta mucho el futbol"

# Tokenizar el texto de entrada
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Cuantización dinámica
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear, torch.nn.LayerNorm}, dtype=torch.qint8
)

# Generar texto adicional basado en la entrada cuantizada
output = quantized_model.generate(
    input_ids,
    max_length=150,  # Ajusta la longitud del texto generado según sea necesario
    num_return_sequences=1,
    no_repeat_ngram_size=2,
    top_k=50,
    top_p=0.95,
    temperature=0.7
)

# Decodificar el texto generado (cuantizado) en español
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Mostrar el texto generado (cuantizado) en español
print("Texto generado (cuantizado):\n", generated_text)

# Aquí es donde puedes cargar y preprocesar tus datos de prueba (reemplaza con tus datos reales)
test_data = [
    "Esta es una oración de prueba 1.",
    "Aquí tienes otra oración de ejemplo.",
    "Puedes agregar más ejemplos de prueba según sea necesario.",
]

# Función de evaluación (lógica ficticia)
def evaluate(model, data):
    total_loss = 0.0
    with torch.no_grad():
        for sentence in data:
            # Tokeniza la oración
            input_ids = tokenizer.encode(sentence, return_tensors="pt")
            # Realiza la inferencia con el modelo cuantizado
            output = quantized_model.generate(input_ids)
            # Supongamos que calculas la pérdida como la diferencia de longitud entre la entrada y el texto generado
            loss = abs(len(output[0]) - len(input_ids[0]))
            # Agrega la pérdida a total_loss
            total_loss += loss

    # Calcula la métrica final (por ejemplo, promedio de la pérdida)
    average_loss = total_loss / len(data)
    return average_loss

# Evaluar el modelo cuantizado
test_loss = evaluate(quantized_model, test_data)
print(f"Pérdida en el conjunto de prueba: {test_loss:.4f}")
```

## Solución del código:
Texto generado (cuantizado):
 Me gusta mucho el futbol, pero no lo hace como un deporte.

El fútbol es un deportes muy popular en la ciudad, y es muy practicado en el barrio de San José. El fútbol se practica en las calles de la zona, en los barrios de La Paz, San Martín, La Merced, El Carmen, Santa Rosa, Los Ángeles, entre otros. La ciudad de Santa Cruz cuenta con un equipo de fútbol profesional, el Club Deportivo Santa Fe, que juega en su estadio municipal, ubicado en San Salvador. En el año 2010, la Asociación de Fútbol de El Salvador (AFSA) anunció que el fútbol en Santa Catarina se organizaría en un torneo de copa, llamado "Clásico de Campeones", que se celebraría cada año

/usr/local/lib/python3.10/dist-packages/transformers/generation/utils.py:1260: UserWarning: Using the model-agnostic default `max_length` (=20) to control the generation length. We recommend setting `max_new_tokens` to control the maximum length of the generation.
  warnings.warn(
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:50256 for open-end generation.
Pérdida en el conjunto de prueba: 11.3333


## Explicación del código:
1-Importamos las bibliotecas necesarias, incluyendo PyTorch y Transformers para GPT-2, junto con las herramientas de cuantización.

2-Cargamos un modelo preentrenado de GPT-2 en español y un tokenizador asociado.

3-Definimos un texto de entrada en español y lo tokenizamos para prepararlo para la generación de texto.

4-Configuramos la cuantización dinámica del modelo. Específicamente, cuantizamos las capas de torch.nn.Linear y torch.nn.LayerNorm. Esto permite que el modelo sea cuantizado, lo que significa que se almacena con una representación de baja precisión para mayor eficiencia.

5-Generamos texto adicional basado en la entrada cuantizada utilizando el modelo cuantizado.

6-Decodificamos el texto generado y lo mostramos en la salida.

7-Preparamos datos de prueba (en este caso, oraciones de prueba ficticias) y definimos una función de evaluación ficticia que calcula la pérdida en función de la diferencia de longitud entre las oraciones de prueba y las oraciones generadas.

8-Evaluamos el modelo cuantizado utilizando nuestros datos de prueba ficticios y mostramos la pérdida en la salida.


## El link del código de google colab es el siguiente:
https://colab.research.google.com/drive/1_agGtx8p9-hjg1vi3qlO10ZDHBhhTu2c






## Análisis y Conclusiones: Compare el rendimiento y el tamaño del modelo original con sus versiones cuantizadas. Reflexione sobre los trade-offs entre tamaño, rendimiento y precisión en modelos de lenguaje. 

Comparación:

Rendimiento: La primera parte del código utiliza el modelo GPT-2 sin cuantización, lo que significa que no se aplica cuantización a los parámetros del modelo. En contraste, la segunda parte del código aplica cuantización dinámica a las capas lineales y de normalización del modelo. La cuantización puede reducir el rendimiento en términos de calidad del texto generado, ya que los parámetros se almacenan en una representación de menor precisión.

Tamaño del modelo: En términos de tamaño del modelo, la versión cuantizada generalmente tiene un tamaño más pequeño que el modelo original, ya que los valores de los parámetros se almacenan con una menor precisión. Esto puede ser beneficioso en términos de almacenamiento y uso de recursos, especialmente en entornos con restricciones de memoria.

Precisión del modelo: La cuantización puede llevar a una pérdida de precisión, especialmente en modelos de lenguaje como GPT-2, donde la generación de texto requiere detalles finos y una alta precisión en las predicciones. La cuantización puede introducir artefactos en el texto generado y afectar la calidad de las respuestas.

En resumen, la cuantización es una técnica que reduce el tamaño del modelo y puede ser útil en entornos con recursos limitados, pero a menudo conlleva una pérdida de precisión y calidad en la generación de texto. La elección entre el modelo original y el modelo cuantizado depende de los trade-offs entre tamaño, rendimiento y precisión que sean aceptables para una aplicación específica.





