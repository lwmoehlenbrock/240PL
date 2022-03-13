import sys

#Different token types
INTEGER, PLUS, MINUS, MULTIPLY, DIVIDE, LPAREN, RPAREN, ID, ASSIGN, SEMI, LBRACKET, RBRACKET, COMMA, EOF = (
    'INTEGER', 'PLUS', 'MINUS', 'MULTIPLY', 'DIVIDE', '(', ')', 'ID', 'ASSIGN', 'SEMI', 'LBRACKET', 'RBRACKET', 'COMMA', 'EOF'
)

#Legal characters for variable names, variables cannot start with numeric characters
acceptable_variable_characters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890_$'

#Used as a flag when parsing assignment statements to know whether the variable is a local variable or a global variable.
inside_Block = False

#Used as a flag to indicate whether an ID token should be parsed as part of an expression or as an assignemnt statement.
assigning = False

class Token(object):
    def __init__(self, type, value):
        self.type = type
        self.value = value

#ID tokens that have a specific meaning in the 240PL grammar
RESERVED_KEYWORDS = {
    'func': Token('func', 'func'),
    'return': Token('return', 'return')
}

#Used to check whether an ID token is part of a function call or just a variable
FUNCTION_NAMES = []

#Creates tokens from the source text of the program being interpreted
class Tokenizer(object):
    def __init__(self, text):
        self.source_text = text
        self.text_index = 0
        self.current_character = self.source_text[self.text_index]

    def error(self):
        raise Exception('Invalid character')

    #Adds subsequent alphanumeric characters to a single string and makes an ID token from the result
    def process_variable_text(self):
        result = ''
        while self.current_character in acceptable_variable_characters:
            result = result + self.current_character
            self.next_character()

        token = RESERVED_KEYWORDS.get(result, Token(ID, result))
        return token


    def next_character(self):
        self.text_index += 1
        if self.text_index > len(self.source_text) - 1:
            self.current_character = None  # end of source code file
        else:
            self.current_character = self.source_text[self.text_index]


    def skip_spaces(self):
        while self.current_character is not None and (self.current_character.isspace() or self.current_character == '\t' or self.current_character == '\n' or self.current_character == '\r'):
            self.next_character()

    #Creates an integer from multiple subsequent numeric characters
    def process_integer(self):
        result = ''
        while self.current_character.isdigit():
            result += self.current_character
            self.next_character()
        return int(result)

    #Breaks the source text up into tokens based on the value of each character in the source text
    def get_next_token(self):

        while self.current_character is not None:

            if self.current_character.isalpha():
                return self.process_variable_text()

            if self.current_character == '=':
                self.next_character()
                return Token(ASSIGN, '=')

            if self.current_character == ';':
                self.next_character()
                return Token(SEMI, ';')
            
            if self.current_character == ',':
                self.next_character()
                return Token(COMMA, ',')

            if self.current_character == '{':
                self.next_character()
                return Token(LBRACKET, '{')

            if self.current_character == '}':
                self.next_character()
                return Token(RBRACKET, '}')
            
            if self.current_character.isspace() or self.current_character == '\t' or self.current_character == '\n' or self.current_character == '\r':
                self.skip_spaces()
                continue

            if self.current_character.isdigit():
                return Token(INTEGER, self.process_integer())

            if self.current_character == '+':
                self.next_character()
                return Token(PLUS, '+')

            if self.current_character == '-':
                self.next_character()
                return Token(MINUS, '-')

            if self.current_character == '*':
                self.next_character()
                return Token(MULTIPLY, '*')

            if self.current_character == '/':
                self.next_character()
                return Token(DIVIDE, '/')

            if self.current_character == '(':
                self.next_character()
                return Token(LPAREN, '(')

            if self.current_character == ')':
                self.next_character()
                return Token(RPAREN, ')')

            #If this function hasn't returned before reaching this point, the current character doesn't match any of the above, then it is an invalid character.
            self.error()

        #If the value of self.current_character is None, then we've reached the end of the source file
        return Token(EOF, None)

#Gets tokens from the tokenizer and constructs an abstract syntax tree from the tokens
class Parser(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # initializes current token to the first token taken from the input
        self.current_token = self.tokenizer.get_next_token()

    def error(self):
        raise Exception('Invalid syntax')

    #After a token is parsed, we replace it with the next token in the input text
    def pop(self, token_type):
        #We need to make sure that the token that is being replaced is the token that was expected.
        #If it doesn't match then the source program has invalid syntax so we throw an error
        if self.current_token.type == token_type:
            self.current_token = self.tokenizer.get_next_token()
        else:
            self.error()

    #Get the value of the token following the current token without replacing the current token. Used to check if an 
    #ID token is a variable or part of a function call or if a function is being assigned an expression or a block
    def peekNextToken(self):
        current_token = self.current_token
        current_index = self.tokenizer.text_index
        current_character = self.tokenizer.current_character
        next_token = self.tokenizer.get_next_token()
        self.current_token = current_token
        self.tokenizer.text_index = current_index
        self.tokenizer.current_character = current_character
        return next_token

    #Recursively build an expression subtree. This enforces the order of operations, the binary operator node made in this function has the lowest priority in the expression (i.e. addition or subtraction)
    def parse_expression(self):
        node = self.parse_term()
        while self.current_token.type == PLUS or self.current_token.type == MINUS:
            token = self.current_token
            if self.current_token.type == PLUS:
                self.pop(PLUS)
            elif self.current_token.type == MINUS:
                self.pop(MINUS)
            
            node = BinaryOperator(node, token, self.parse_term())

        return node

    #The binary operator node made in this function has the next highest priority from that made in the previous function (i.e. multiplication or division)
    def parse_term(self):
        node = self.parse_factor()
        while self.current_token.type == MULTIPLY or self.current_token.type == DIVIDE:
            token = self.current_token
            if self.current_token.type == MULTIPLY:
                self.pop(MULTIPLY)
            elif self.current_token.type == DIVIDE:
                self.pop(DIVIDE)

            node = BinaryOperator(node, token, self.parse_factor())
        return node

    #The node made in this function has the highest priority in the order of operations, for example a variable or an integer or an expression contained within parentheses.
    def parse_factor(self):
        #indicates unary minus (i.e making a number negative)
        if self.current_token.type == MINUS:
            self.pop(MINUS)
            return UnaryOperator(MINUS, self.parse_factor())
        #Since we have unary minus, might as well include unary plus
        elif self.current_token.type == PLUS:
            self.pop(PLUS)
            return UnaryOperator(PLUS, self.parse_factor())
        elif self.current_token.type == INTEGER:
            token = self.current_token
            self.pop(INTEGER)
            return Integer(token)
        elif self.current_token.type == LPAREN:
            self.pop(LPAREN)
            node = self.parse_expression()
            self.pop(RPAREN)
            return node
        else:
            variable_name = self.variable()
            functions = []
            for tuple in FUNCTION_NAMES:
                functions.append(tuple[0])
            #If the variable name is in the list of defined functions, a function call is being made, otherwise it is just a normal variable
            if variable_name.value in functions:
                self.pop(LPAREN)
                node = self.function_call(variable_name)
            else:    
                node = variable_name
            return node

    #Recursively build the abstract syntax tree by creating a node for each statement in the program and create the root node.
    def statement_list(self):
        #Each program must have at least one statement so process that statement first
        node = self.statement()

        #This list stores each statement of the program, and each node in the list is a child of the root node
        results = [node]

        #Statements are separated by semicolons, so a semicolon token indicates that there is another statement to parse
        #It should be noted that the last statement in a program does not have to have a semicolon after it.
        #If it is followed by a semicolon, then the result of self.statement will be self.empty.
        while self.current_token.type == SEMI:
            self.pop(SEMI)
            results.append(self.statement())

        if self.current_token.type == ID:
            self.error()

        return StatementList(results)

    #This function determines what type of statement the current token represents and creates the appropriate node for the abstract syntax tree.
    def statement(self):
        #indicates whether we are parsing statements inside a function block, if so we want to look in the local scope instead of the global scope
        global inside_Block
        global assigning
        if self.current_token.type == 'func':
            node = self.function_assignment()
        elif self.current_token.type == 'return':
            node = self.return_statement()
        elif self.current_token.type == ID:
            if assigning:
                node = self.parse_expression()
            else:
                if not inside_Block:
                    variable_name = self.variable()
                else:
                    variable_name = self.local_variable()
                if self.current_token.type == LPAREN:
                    self.pop(LPAREN)
                    node = self.function_call(variable_name)
                else:    
                    assigning = True
                    node = self.assignment_statement(variable_name)
                    assigning = False
        else:
            node = self.empty()
        return node

    #For returning a value from a block function
    def return_statement(self):
        self.pop('return')
        node = self.parse_expression()
        return Return(node)


    def function_call(self, function_name):
        paramaters = []
        while self.current_token.type != RPAREN:
            if self.current_token.type == COMMA:
                self.pop(COMMA)
            if self.peekNextToken().type == LPAREN:
                local_variable_flag = False
                for tuple in FUNCTION_NAMES:
                    if function_name == tuple[0]:
                        if self.current_token.value in tuple[1]:
                            local_variable_flag = True
                            variable_name = self.local_variable()
                if not local_variable_flag:
                    variable_name = self.variable()
                self.pop(LPAREN)
                paramaters.append(self.function_call(variable_name))
            else:
                paramaters.append(self.parse_expression())
        self.pop(RPAREN)
        left = function_name
        right = paramaters
        node = FunctionCall(left, right)
        return node

    def assignment_statement(self, variable):
        global inside_Block
        left = variable
        token = self.current_token
        self.pop(ASSIGN)
        right = self.parse_expression()
        
        #inside_Block indicates whether the assigned variable should be a local or global variable
        if not inside_Block:
            node = Assign(left, token, right)
        else:
            node = BlockAssign(left, token, right)
        return node

    def function_assignment(self):
        self.pop('func')
        left = self.variable()
        
        self.pop(LPAREN)
        paramater_list = []
        while self.current_token.type != RPAREN:
            if self.current_token.type == COMMA:
                self.pop(COMMA)
            paramater_list.append(self.local_variable())
            
        self.pop(RPAREN)
        token = self.current_token
        self.pop(ASSIGN)
        if self.current_token.type == LBRACKET:
            right = self.block()
        else:
            right = self.parse_expression()
        FUNCTION_NAMES.append((left.value,paramater_list))
        node = FunctionAssign(left, token, right, paramater_list)
        return node
        

    def block(self):
        global inside_Block
        inside_Block = True
        self.pop(LBRACKET)
        node = self.statement()

        results = [node]

        while self.current_token.type == SEMI:
            self.pop(SEMI)
            results.append(self.statement())

        if self.current_token.type == ID:
            self.error()

        self.pop(RBRACKET)
        inside_Block = False
        return Block(results)


    def variable(self):
        node = Var(self.current_token)
        self.pop(ID)
        return node

    def local_variable(self):
        node = LocalVar(self.current_token)
        self.pop(ID)
        return node

    def empty(self):
        return NoOp()


    def parse(self):
        #Start the process of recursively building the abstract syntax tree
        node = self.statement_list()
        #After parsing the statement list, the program should be fully parsed, if there are any more tokens besides EOF then something went wrong (i.e there was invalid syntax)
        if self.current_token.type != EOF:
            self.error()

        return node

#Base object that each node class inherits from
class AST(object):
    pass

#Each of these classes represents a node type of the abstract syntax tree
class BinaryOperator(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class UnaryOperator(AST):
    def __init__(self, op, expression):
        self.token = self.op = op
        self.expression = expression

class Block(AST):
    def __init__(self, node):
        self.children = node
        self.local_scope = {}

class StatementList(AST):
    def __init__(self, node):
        self.children = node

class BlockAssign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right

class Assign(AST):
    def __init__(self, left, op, right):
        self.left = left
        self.token = self.op = op
        self.right = right
    
class FunctionAssign(AST):
    def __init__(self, left, op, right, paramater_list):
        self.left = left
        self.token = self.op = op
        self.right = right
        self.paramater_list = paramater_list

class Function(AST):
    def __init__(self, left, op, right, paramater_list):
        self.left = left
        self.token = self.op = op
        self.right = right
        self.paramater_list = paramater_list


class FunctionCall(AST):
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.local_scope = {}

class Var(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

class LocalVar(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value
    
class NoOp(AST):
    pass

class Return(AST):
    def __init__(self, node):
        self.node = node

class Integer(AST):
    def __init__(self, token):
        self.token = token
        self.value = token.value

#Class for traversing the abstract syntax tree. Interpreter will inherit from this class
class NodeVisitor(object):
    #Wrapper function that determines the type of a node and calls the appropriate visitor function in the interpreter class
    def visit(self, node):
        method_name = 'visit_' + type(node).__name__
        visitor = getattr(self, method_name, self.generic_visit)
        return visitor(node)

#Class that is responsible for traversing the abstract syntax tree made by the Parser and evaluating the program
class Interpreter(NodeVisitor):

    def __init__(self, parser):
        self.parser = parser
        self.GLOBAL_SCOPE = {}
        #used to store the current local scope for nested function calls
        self.store_local_scope = None
        self.local_scope = None
    
    def visit_StatementList(self, node):
        for child in node.children:
            self.visit(child)
            if self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None

    def visit_Block(self, node):
        global inside_Block
        inside_Block = True
        for child in node.children:
            if type(child).__name__ == 'Return':
                evaluation = self.visit(child)
                inside_Block = False
                return evaluation
            else:
                if type(child) == BlockAssign:
                    self.visit(child) 
                    if self.store_local_scope is not None:
                        self.local_scope = self.store_local_scope
                        self.store_local_scope = None
                else:
                    function_call = False
                    if type(child).__name__ == 'FunctionCall':
                        function_call = True
                    self.visit(child)
                    if function_call and self.store_local_scope is not None:
                        self.local_scope = self.store_local_scope
                        self.store_local_scope = None
        inside_Block = False

    def visit_Return(self, node):
        function_call = False
        if type(node.node).__name__ == 'FunctionCall':
            function_call = True
        return_value = self.visit(node.node)
        if function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
        return return_value

    def visit_NoOp(self, node):
        pass

    def visit_Assign(self, node):
        var_name = node.left.value
        assigned_value = self.visit(node.right)
        if self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
        self.GLOBAL_SCOPE[var_name] = assigned_value

    def visit_BlockAssign(self, node):
        var_name = node.left.value
        assigned_value = self.visit(node.right)
        if self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
        self.local_scope[var_name] = assigned_value

    def visit_FunctionAssign(self, node):
        global inside_Block
        function_name = node.left.value
        function_node = Function(node.left, node.op, node.right, node.paramater_list)
        if inside_Block:
            self.local_scope[function_name] = function_node
        else:
            self.GLOBAL_SCOPE[function_name] = function_node

    def visit_FunctionCall(self, node):
        global inside_Block
        if self.local_scope is not None:
            self.store_local_scope = self.local_scope
        function_name = node.left.value
        function_paramaters = node.right
        if inside_Block:
            function_body = self.local_scope[function_name]
        else:
            function_body = self.GLOBAL_SCOPE[function_name]
        for passed_paramater, paramater in zip(function_paramaters, function_body.paramater_list):
            node.local_scope[paramater.value] = self.visit(passed_paramater)
        self.local_scope = node.local_scope
        return self.visit(function_body.right)



    def visit_LocalVar(self, node):
        var_name = node.value
        val = self.local_scope.get(var_name)
        if val is None:
            raise NameError(repr(var_name))
        else:
            return val

    def visit_Var(self, node):
        var_name = node.value
        for key in self.local_scope:
            if var_name == key:
                return self.visit_LocalVar(node)
        val = self.GLOBAL_SCOPE.get(var_name)
        if val is None:
            raise NameError(repr(var_name))
        else:
            return val

    def visit_BinaryOperator(self, node):
        left_function_call = False
        right_function_call = False
        if type(node.left).__name__ == 'FunctionCall':
            left_function_call = True
        if type(node.right).__name__ == 'FunctionCall':
            right_function_call = True

        if node.op.type == PLUS:
            left_term = self.visit(node.left)
            if left_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            right_term = self.visit(node.right)
            if right_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            return left_term + right_term
        elif node.op.type == MINUS:
            left_term = self.visit(node.left)
            if left_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            right_term = self.visit(node.right)
            if right_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            return left_term - right_term
        elif node.op.type == MULTIPLY:
            left_term = self.visit(node.left)
            if left_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            right_term = self.visit(node.right)
            if right_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            return left_term * right_term
        elif node.op.type == DIVIDE:
            left_term = self.visit(node.left)
            if left_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            right_term = self.visit(node.right)
            if right_function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            return int(left_term / right_term)

    def visit_UnaryOperator(self, node):
        function_call = False
        if type(node.expression).__name__ == 'FunctionCall':
            function_call = True
        if node.op == PLUS:
            term = self.visit(node.expression)
            if function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            return +term
        elif node.op == MINUS:
            term = self.visit(node.expression)
            if function_call and self.store_local_scope is not None:
                self.local_scope = self.store_local_scope
                self.store_local_scope = None
            return -term
    def visit_Integer(self, node):
        return node.value

    def interpret(self):
        tree = self.parser.parse()
        return self.visit(tree)



def main():
    with open(sys.argv[1]) as f:
        lines = f.readlines()

    text = ''
    for line in lines:
        text = text + line

    tokenizer = Tokenizer(text)
    parser = Parser(tokenizer)
    interpreter = Interpreter(parser)
    interpreter.interpret()
    print(interpreter.GLOBAL_SCOPE)


if __name__ == '__main__':
    main()