# Basics of Python Syntax

# Indentation
if True:
    print("This block is indented, representing proper codestructure")

    # Variable declaration and assignment
    variable = 10
    Variable = 20
    print(variable)
    print(Variable)

    # Dynamic typing
    dynamic_variable = 3.14
    dynamic_variable = "Hello"
    print(dynamic_variable)

    # Built-in functions
    text_length = len("Python syntax")
    print(text_length)

    user_input = input("Enter something: ")
    print("Text entered:", user_input)

    # Conditional statements
    if text_length > 10:
        print("Text length is greater than 10")
    elif text_length == 10:
        print("Text length is exactly 10")
    else:
        print("Text length is less than 10")

    # Loops
    for i in range(3):
        print("Iteration:", i)

while text_length > 0:
    print("Text length:", text_length)
    text_length -= 1
