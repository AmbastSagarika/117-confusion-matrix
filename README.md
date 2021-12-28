# 117-confusion-matrix



factors = df[["age", "sex", "cp", "chol", "thalach"]]
heart_attack = df["target"]

factors_train, factors_test, heart_attack_train, heart_attack_test = train_test_split(factors, heart_attack, test_size = 0.25, random_state = 0)




from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()

factors_train = sc_x.fit_transform(factors_train)  
factors_test = sc_x.transform(factors_test)





classifier2 = LogisticRegression(random_state = 0) 
classifier2.fit(factors_train, heart_attack_train)





heart_attack_prediction_1 = classifier2.predict(factors_test)

predicted_values_1 = []
for i in heart_attack_prediction_1:
  if i == 0:
    predicted_values_1.append("No")
  else:
    predicted_values_1.append("Yes")

actual_values_1 = []
for i in heart_attack_test.ravel():
  if i == 0:
    actual_values_1.append("No")
  else:
    actual_values_1.append("Yes")
    
    
    
    
    
    
    
    
    
cm = confusion_matrix(actual_values_1, predicted_values_1, labels)

ax= plt.subplot()
sns.heatmap(cm, annot=True, ax = ax)

ax.set_xlabel('Predicted')
ax.set_ylabel('Actual') 
ax.set_title('Confusion Matrix')
ax.xaxis.set_ticklabels(labels); ax.yaxis.set_ticklabels(labels)
