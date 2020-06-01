domain_list = ['art', 'clipart', 'product', 'real_world']

for i in domain_list:
    for j in domain_list:
        if i != j:
            print(i, '-->', j)
