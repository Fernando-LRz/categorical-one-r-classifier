import pandas

def main():
    # Dataset file
    csv = '../golf-dataset-categorical.csv'

    # Leer el archivo CSV y almacenar los datos en un DataFrame
    df = pandas.read_csv(csv)

    # Eliminar los espacios en blanco del DataFrame
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Verificar si el script se está ejecutando como programa principal
if __name__ == '__main__':
    main()
    