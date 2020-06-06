import numpy as np



def ingredient_recommendation(index,d,Vectorizer):
     
    """
    Recommend ingredients basd on a list of ingredients. 
    Recommended ingredients will be print with their recommendation indices in a descending order, meaning
    ingredients we recommend more will be printed first
    
    Arguments:
        index: a list of number corresponds to the index of input ingredients
        d: returned dictionary of function vectorizor_preparation(bow_corpus),containing
           dtm: Document-Term matrix
           norm_dtm: normalized Document-Term matrix
           tfidf: TF-IDF matrix
           norm_tfidf: normalized TF-IDF matrix
           p: number of different words in total
        Vectorizer: a string specifying which vectorizer to use, can be "Count" or "Tfidf"
    
    Return:
        recommendation: a dictionary of recommended ingredients with values being their recommendation index. 
        The larger the recommendation index is, the more we recommend an ingredient.
            
    """
    
    # prepare the vectorizer
    p=d["p"]
    ingredients=d["ingredients"]
    dtm=d["dtm"]
    norm_dtm=d["norm_dtm"]
    tfidf=d["tfidf"]
    norm_tfidf=d["norm_tfidf"]
    
    # create "recipe" vector from index
    recipe=np.zeros((p,1))
    recipe[index]=1
        
    # show the ingredients of "recipe"
    print("The ingredients we have are:","\n")
    for i in index:
        print(ingredients[i])
    
    # choose vectorizer:
    if Vectorizer == "Count":
        # normalize the "recipe" vector
        recipe=recipe/np.linalg.norm(recipe)
        
        # calculate the similarity between "recipe" and existing recipes
        similarity=np.dot(norm_dtm,recipe)

        # find the most similar recipe
        loc = np.argmax(similarity)
        
        print("\n")
        print("The similarity socre of our recommendation is:","\n",similarity[loc])
        rec_index=np.nonzero(norm_dtm[loc])
        
        # calculate the recommendation index
        recommendation={}
        for i in rec_index[0]:
            if i not in index:
                row=np.nonzero(dtm[:,i])[0]
                recommendation[ingredients[i]]=sum(np.sum(dtm[row,:],axis=0)[index])

        # sort the recommended ingredients by their index
        print("\n")
        print("The recommended ingredients are:","\n")
        for key, value in sorted(recommendation.items(), key=lambda item: item[1],reverse = True):
            print("%s: %s" % (key, value))
        return(recommendation)
    
              
    if Vectorizer == "Tfidf":
        # calculate the tf-idf for the "recipe" vector and normalize it
        temp=np.log(np.sum(dtm)/np.sum(dtm,axis=0)).reshape(p,1)
        recipe[index]=temp[index]
        recipe=recipe/np.linalg.norm(recipe)
            
        # calculate the similarity between "recipe" and existing recipes
        similarity=np.dot(norm_tfidf,recipe)

        # find the most similar recipe
        loc = np.argmax(similarity)
        
        print("\n")
        print("The similarity socre of our recommendation is:","\n",similarity[loc])
        rec_index=np.nonzero(norm_tfidf[loc])
        
        # calculate the recommendation index
        recommendation={}
        for i in rec_index[0]:
            if i not in index:
                row=np.nonzero(dtm[:,i])[0]
                recommendation[ingredients[i]]=sum(np.sum(dtm[row,:],axis=0)[index])

        # sort the recommended ingredients by their index
        print("\n")
        print("The recommended ingredients are:","\n")
        for key, value in sorted(recommendation.items(), key=lambda item: item[1],reverse = True):
            print("%s: %s" % (key, value))
        return(recommendation)