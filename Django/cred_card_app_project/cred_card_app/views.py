from django.http import HttpResponse
from django.shortcuts import render
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

dataset=pd.read_csv('X_dataset.csv')

def home(request):
    return render(request,'home2.html')



def result(request):
    cls= joblib.load('final_model_random.sav')
    lis=[]
    lis.append(0)
    lis.append(request.POST.get('input1')[:1])
    lis.append(request.POST.get('input2'))
    lis.append(request.POST.get('input3'))
    lis.append(0)
    lis.append(int(request.POST.get('input4')))
    lis.append(request.POST.get('input14'))
    lis.append(request.POST.get('input5'))
    lis.append(request.POST.get('input6'))
    lis.append(request.POST.get('input7'))
    lis.append(int(request.POST.get('input8')))
    lis.append(int(request.POST.get('input9')))
    lis.append(0)
    lis.append(int(request.POST.get('input10'))) 
    lis.append(int(request.POST.get('input11')))
    lis.append(int(request.POST.get('input12')))
    lis.append(0)
    lis.append(int(request.POST.get('input13')))
    lis.append(0)

    x_test = pd.DataFrame(data=[lis],columns=dataset.columns)
    train_x= pd.concat([dataset,x_test],ignore_index=True)
    X_trn=transform(train_x)
    rec=X_trn[X_trn["ID"]==0].drop(columns='ID')
    print(rec)
    prediction = cls.predict(rec)
    print(prediction)
    approved = False
    if prediction == 0:
        approved=True
    #x_h=rec.to_html()
    #return render(x_h)
    return render(request,'result.html',{'approved':approved})

def transform(df):
    X_data=df
    #dropping columns
    X_data.drop(['Account age','FLAG_MOBIL','CNT_CHILDREN','OCCUPATION_TYPE'],axis=1,inplace=True)
    #encoding categorical features
    cat_col = ['CODE_GENDER','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','FLAG_WORK_PHONE','FLAG_PHONE','FLAG_EMAIL']
    dummies=pd.get_dummies(X_data[cat_col],drop_first=True)
    X_data.drop(cat_col,axis=1,inplace=True)
    X_data=pd.concat([dummies,X_data],axis=1)
    #Handling Outliers
    percentiles = X_data['CNT_FAM_MEMBERS'].quantile([0.05,0.99]).values
    X_data['CNT_FAM_MEMBERS'][X_data['CNT_FAM_MEMBERS'] <= percentiles[0]] = percentiles[0]
    X_data['CNT_FAM_MEMBERS'][X_data['CNT_FAM_MEMBERS'] >= percentiles[1]] = percentiles[1]
    percentiles = X_data['AMT_INCOME_TOTAL'].quantile([0.05,0.99]).values
    X_data['AMT_INCOME_TOTAL'][X_data['AMT_INCOME_TOTAL'] <= percentiles[0]] = percentiles[0]
    X_data['AMT_INCOME_TOTAL'][X_data['AMT_INCOME_TOTAL'] >= percentiles[1]] = percentiles[1]
    percentiles = X_data['EMPLOYMENT_LENGTH'].quantile([0.05,0.99]).values
    X_data['EMPLOYMENT_LENGTH'][X_data['EMPLOYMENT_LENGTH'] <= percentiles[0]] = percentiles[0]
    X_data['EMPLOYMENT_LENGTH'][X_data['EMPLOYMENT_LENGTH'] >= percentiles[1]] = percentiles[1]
    #Scaling numerical features
    scaler = MinMaxScaler()
    num_cat=['AGE','AMT_INCOME_TOTAL','EMPLOYMENT_LENGTH','CNT_FAM_MEMBERS']
    X_data[num_cat]=scaler.fit_transform(X_data[num_cat])
    return X_data