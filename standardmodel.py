import json

def read_data(file_name):
    with open(file_name) as data_file:    
        data = json.load(data_file)
    list_question=[]
    equation=[]
    solution=[]
    for i in range(len(data)):
        list_question.append(data[i]['sQuestion'])
        equation.append(data[i]['lEquations'])
        solution.append(data[i]['lSolutions'])
    return list_question,solution
def process_equation(equation):
    operation=[]
    add=0
    sub=0
    mul=0
    div=0
    for i in equation:
        if '+' in i[0]:
            add=add+1
            operation.append('+')
        elif '-' in i[0]:
            sub=sub+1
            operation.append('-')
        elif '*'in i[0]:
            mul=mul+1
            operation.append('*')
        elif '/' in i[0]:
            div=div+1
            operation.append('/')
        else:
            print("hey")
            print(i)
    print("add"+str(add))
    print("sadd"+str(sub))
    print("madd"+str(mul))
    print("d"+str(div))
    return operation


def chunck_question(question):
    '''Takes out question part from the whole question
    '''
    question_word=["How","When","What","Find","Calculate","how"]
    tokens = [token.lower() for token in question_word]
    p=0
    list_q=[]
    query=[]
    for i in question:
        current_question=[]
        current_query=[]
        for j in i.split():
            if (len(current_query)==0):
                if j in question_word:
                    current_query.append(j)
                else:
                    current_question.append(j)
            elif(len(current_query)>0):
                current_query.append(j)
        cq=" ".join(current_question)
        cque=" ".join(current_query)
        list_q.append(cq)
        query.append(cque)
    return list_q,query

list_question,equation=read_data('DATA/addsub.json')
que=["Mary loves eating fruits . Mary paid $11.08 for berries , $14.33 for apples , and $9.31 for peaches . In total , how much money did she spend ?"]
worldstate,query=chunck_question(que)
print(query)
print(len(worldstate),len(query))
vocab=[]
for i in query:
    print(i[0])
    for j in query:
        print(j)
        for q in j.split(): 
            print(q)
            vocab.append(q)
