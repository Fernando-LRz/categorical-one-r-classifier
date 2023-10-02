import pandas

def main():
    # Archivo con el set de datos
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
        frecuency_table = pandas.crosstab(data[attribute], data[class_name]) # filas, columnas
        frequency_tables[attribute] = frecuency_table

    # Crear un DataFrame para almacenar las reglas
    rules = pandas.DataFrame(columns=['Attribute', 'Value', 'Class'])

    # Iterar sobre las tablas de frecuencia 
    for attribute, frecuency_table in frequency_tables.items():
        # Obtener los valores del atributo
        attribute_values = frecuency_table.index

        # Iterar sobre los nombres de los atributos
        for value in attribute_values:
            # Encontrar la clase más frecuente para el valor del atributo
            most_frequent_class = data[data[attribute] == value][class_name].mode().iloc[0]
            rules = rules._append({'Attribute': attribute, 'Value': value, 'Class': most_frequent_class}, ignore_index=True)
    
    # Agregar una columna para contabilizar los errores
    rules['Errors'] = 0

    # Agregar una columna para contabilizar las instancias
    rules['Instances'] = 0
    
    # Iterar sobre las reglas
    for index, regla in rules.iterrows():
        # Obtener los datos de cada regla
        attribute = regla['Attribute']
        value = regla['Value']
        expected_class = regla['Class']

        # Obtener y asignar la cantidad de instancias cuyo valor del atributo coincide con el valor del atributo de la regla
        instances = len(data[data[attribute] == value])
        rules.at[index, 'Instances'] = instances

        # Obtener y asignar la cantidad de instancias que no coinciden con la regla
        errors = len(data[(data[attribute] == value) & (data[class_name] != expected_class)])
        rules.at[index, 'Errors'] = errors
    
    # Crear un DataFrame para almacenar los errores totales
    total_errors = pandas.DataFrame(columns=['Attribute', 'Errors', 'Instances'])

    # Iterar sobre los nombres de los atributos 
    for attribute in attributes:
        # Obtener las reglas del atributo
        attribute_rule = rules[rules['Attribute'] == attribute]
        
        # Sumar los errores de las reglas
        errors = attribute_rule['Errors'].sum()

        # Sumar las instancias 
        instances = attribute_rule['Instances'].sum()
        
        # Guardar los errores y las instancias de cada regla
        total_errors = total_errors._append({'Attribute': attribute, 'Errors': errors, 'Instances': instances}, ignore_index=True)
    
    # Calcular la división Errors/Instances, que representa el error total
    total_errors['Result'] = total_errors['Errors'] / total_errors['Instances']

    # Encontrar el índice del atributo cuya regla es la de menor error total
    selected_rule_index = total_errors['Result'].idxmin()  # Si es más de una regla, retornar la primera

    # Obtener el nombre del atributo cuya regla es la de menor error total
    selected_rule_name = total_errors.loc[selected_rule_index]['Attribute']

    # Obtener la regla
    selected_rule = rules[rules['Attribute'] == selected_rule_name]

    # Crear una lista para almacenar los resultados
    result = []

    # Iterar sobre cada instancia en el conjunto de datos
    for instance_index, instance in data.iterrows():
        # Obtener de la instancia el valor del atributo que se evalúa en la regla y también su clase
        instance_attribute_value = instance[selected_rule_name]
        instance_class = instance[class_name]

        # Crear una bandera para saber si la regla acertó
        rule_fulfilled = False

        # Obtener la clase esperada según la regla
        expected_class = (selected_rule[selected_rule['Value'] == instance_attribute_value]['Class']).values[0]

        # Iterar sobre cada condición en la regla seleccionada
        for condition_index, condition in selected_rule.iterrows():
            # Obtener de la condicion el valor del atributo y también su clase
            condition_attribute_value = condition['Value']

            # Evaluar si la regla acierta
            if instance_attribute_value == condition_attribute_value and instance_class == expected_class:            
                rule_fulfilled = True
                break
 
        result.append([expected_class, rule_fulfilled])
    
    # Convertir la lista de resultados en un DataFrame
    result_df = pandas.DataFrame(result, columns=['Clase Esperada', 'Acierto'])

    # Combinar el dataframe de los datos y el de resultados
    final_df = pandas.concat([data, result_df], axis=1)


# Ejecutar el main
if __name__ == '__main__':
    main()   