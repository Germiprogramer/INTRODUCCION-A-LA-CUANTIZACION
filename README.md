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

