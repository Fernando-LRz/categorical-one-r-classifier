import pandas
from fractions import Fraction

def main():
    # Dataset file
    csv = '../golf-dataset-categorical.csv'

    # Leer el archivo CSV y almacenar los datos en un DataFrame
    data = pandas.read_csv(csv)

    # Eliminar los espacios en blanco del DataFrame
    data = data.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Obtener una lista con los nombres de los atributos (omitir la última columna)
    attributes = data.columns[:-1]

    # Obtener el nombre de la clase
    class_name = data.columns[-1]

    # Crear un diccionario para almacenar las tablas de frecuencia
    frequency_tables = {}

    # Iterar sobre cada atributo y calcular su tabla de frecuencia
    for attribute in attributes:
        frecuency_table = pandas.crosstab(data[attribute], data[class_name]) # rows, columns
        frequency_tables[attribute] = frecuency_table

    # Crear un DataFrame para almacenar las reglas
    rules = pandas.DataFrame(columns=['Attribute', 'Value', 'Class'])

    # Iterar sobre las tablas de frecuencia 
    for attribute, frecuency_table in frequency_tables.items():
        # Obtener los valores del atributo
        attribute_values = frecuency_table.index

        for value in attribute_values:
            # Encontrar la clase más frecuente para el valor del atributo
            most_frequent_class = data[data[attribute] == value][class_name].mode().iloc[0]
            rules = rules._append({'Attribute': attribute, 'Value': value, 'Class': most_frequent_class}, ignore_index=True)
    
    # Agregar una columna para contabilizar los errores
    rules['errors'] = 0

    # Agregar una columna para contabilizar las instancias
    rules['instances'] = 0
    
    # Iterar sobre las reglas
    for index, regla in rules.iterrows():
        # Obtener los datos de cada regla
        attribute = regla['Attribute']
        value = regla['Value']
        expected_class = regla['Class']

        # Obtener y asignar la cantidad de instancias cuyo valor del atributo coincide con el valor del atributo de la regla
        instances = len(data[data[attribute] == value])
        rules.at[index, 'instances'] = instances

        # Obtener y asignar la cantidad de instancias que no coinciden con la regla
        errors = len(data[(data[attribute] == value) & (data[class_name] != expected_class)])
        rules.at[index, 'errors'] = errors

    print(rules)

# Ejecutar el main
if __name__ == '__main__':
    main()
    