import pandas

def main():
    # Dataset file
    csv = '../golf-dataset-categorical.csv'

    # Leer el archivo CSV y almacenar los datos en un DataFrame
    df = pandas.read_csv(csv)

    # Eliminar los espacios en blanco del DataFrame
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Obtener una lista con los nombres de los atributos (omitir la última columna)
    attributes = df.columns[:-1]

    # Obtener el nombre de la clase
    class_name = df.columns[-1]

    # Crear un diccionario para almacenar las tablas de frecuencia
    frequency_tables = {}

    # Iterar sobre cada atributo y calcular su tabla de frecuencia
    for attribute in attributes:
        frecuency_table = pandas.crosstab(df[attribute], df[class_name]) # rows, columns
        frequency_tables[attribute] = frecuency_table

    # Crear un DataFrame para almacenar las reglas
    rules = pandas.DataFrame(columns=['Attribute', 'Value', 'Class'])

    # Iterar sobre las tablas de frecuencia 
    for attribute, frecuency_table in frequency_tables.items():
        # Obtener los valores posibles de cada atributo
        attribute_values = frecuency_table.index

        for value in attribute_values:
            # Encuentra la clase más frecuente para el valor del atributo
            most_frequent_class = df[df[attribute] == value][class_name].mode().iloc[0]
            rules = rules._append({'Attribute': attribute, 'Value': value, 'Class': most_frequent_class}, ignore_index=True)
    
    print(rules)

# Verificar si el script se está ejecutando como programa principal
if __name__ == '__main__':
    main()
    