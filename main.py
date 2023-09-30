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

    # Imprimir las tablas de frecuencia por atributo
    for attribute, tabla in frequency_tables.items():
        print(f"Tabla de frecuencia para el atributo '{attribute}':")
        print(tabla)
        print()

# Verificar si el script se está ejecutando como programa principal
if __name__ == '__main__':
    main()
    