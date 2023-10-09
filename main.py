import pandas
from OneR import OneR

def main():
    # Archivo con el set de datos
    csv = '../golf-dataset-categorical.csv'

    # Leer el archivo CSV y almacenar los datos en un DataFrame
    dataset = pandas.read_csv(csv)

    # Eliminar los espacios en blanco del DataFrame
    dataset = dataset.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Definir el tamaño del set de datos de entrenamiento
    training_percentage = 0.7
    number_of_instances = round(len(dataset) * training_percentage)
    
    # Definir el set de datos de entrenamiento, seleccionando aleatoriamente las instancias
    training_dataset = dataset.sample(number_of_instances)

    # Definir el set de datos de prueba
    test_dataset = dataset.drop(training_dataset.index)

    # Crear una instancia de la clase OneR
    oneR = OneR(training_dataset)

    oneR.fit()
    oneR.evaluate(test_dataset)

    frequency_tables = oneR.getFrequencyTables()
    rules = oneR.getRules()
    total_errors = oneR.getTotalErrors()
    model = oneR.getModel()
    evaluation = oneR.getModelEvaluationResult()

    print()

    print('Conjunto de datos de entrenamiento')
    print()
    print(training_dataset)
    print()

    print('Conjunto de datos de prueba')
    print()
    print(test_dataset)
    print()

    print('Tablas de frecuencia')
    print()
    for key, value in frequency_tables.items():
        print(value)
        print()

    print('Reglas')
    print()
    print(rules)
    print()

    print('Errores totales')
    print()
    print(total_errors)
    print()

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