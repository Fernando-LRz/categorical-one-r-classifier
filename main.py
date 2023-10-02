import pandas
from OneR_Prediction import OneR_Prediction

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
    oneR_Prediction = OneR_Prediction(training_dataset, test_dataset, attributes, class_name)

    # Calcular las tablas de frecuencia
    oneR_Prediction.calculateFrequencyTables()

    # print(oneR_Prediction.rules)
    # print()

    # Calcular los errores
    oneR_Prediction.calculateErrors()

    # print(oneR_Prediction.rules)
    # print()

    # Calcular el error total de las reglas
    oneR_Prediction.calculateTotalErrors()

    # print(oneR_Prediction.total_errors)
    # print()

    # Seleccionar la regla con el menor error total
    oneR_Prediction.selectTheBestRule()

    # print(oneR_Prediction.selected_rule)
    # print()

    # Evaluar la regla seleccionada
    oneR_Prediction.evaluateSelectedRule()

    print(oneR_Prediction.rules)
    print()
    print(oneR_Prediction.total_errors)
    print()
    print(oneR_Prediction.selected_rule)
    print()
    print(oneR_Prediction.result)
    print()


# Ejecutar el main
if __name__ == '__main__':
    main()   