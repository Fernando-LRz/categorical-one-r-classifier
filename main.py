import pandas
from OneR import OneR

def main():
    # Archivo con el set de datos
    csv = '../golf-dataset-categorical.csv'

    # Leer el archivo CSV y almacenar los datos en un DataFrame
    dataset = pandas.read_csv(csv)

    # Eliminar los espacios en blanco del DataFrame
    dataset = dataset.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Obtener una lista con los nombres de los atributos (omitir la última columna)
    attributes = dataset.columns[:-1]

    # Obtener el nombre de la clase (última columna)
    class_name = dataset.columns[-1]

    # Definir el tamaño del set de datos de entrenamiento
    training_percentage = 0.7
    number_of_instances = round(len(dataset) * training_percentage)
    
    # Definir el set de datos de entrenamiento, seleccionando aleatoriamente las instancias
    training_dataset = dataset.sample(number_of_instances)

    # Definir el set de datos de prueba
    test_dataset = dataset.drop(training_dataset.index)

    # Crear una instancia de la clase OneR_Prediction
    oneR_Prediction = OneR(training_dataset, attributes, class_name)

    oneR_Prediction.fit()
    oneR_Prediction.evaluate(test_dataset)

    model = oneR_Prediction.getModel()
    evaluation = oneR_Prediction.getModelEvaluationResult()

    print()

    print('Conjunto de datos de entrenamiento')
    print()
    print(training_dataset)
    print()

    print('Conjunto de datos de prueba')
    print()
    print(test_dataset)
    print()

    # oneR_Prediction.showAdditionalInformation()

    print('Modelo')
    print()
    print(model)
    print()
    
    print('Evaluación del modelo')
    print()
    print(evaluation)
    print()

# Ejecutar el main
if __name__ == '__main__':
    main()   