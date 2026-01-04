import json

with open('notebooks/eda.ipynb', 'r') as f:
    nb = json.load(f)

# Add hyperparameter tuning for Logistic Regression
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Hyperparameter Tuning for Logistic Regression\n",
        "from sklearn.model_selection import GridSearchCV\n",
        "\n",
        "param_grid_lr = {'C': [0.01, 0.1, 1, 10, 100]}\n",
        "grid_lr = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000), param_grid_lr, cv=5, scoring='recall')\n",
        "grid_lr.fit(X_train, y_train)\n",
        "\n",
        "best_lr = grid_lr.best_estimator_\n",
        "print(\"Best params for Logistic Regression:\", grid_lr.best_params_)\n",
        "\n",
        "# Evaluate tuned model\n",
        "y_pred_lr_tuned = best_lr.predict(X_test)\n",
        "recall_lr_tuned = recall_score(y_test, y_pred_lr_tuned)\n",
        "precision_lr_tuned = precision_score(y_test, y_pred_lr_tuned)\n",
        "print(f\"Tuned LR - Recall: {recall_lr_tuned:.3f}, Precision: {precision_lr_tuned:.3f}\")\n"
    ]
})

# Add hyperparameter tuning for Decision Tree
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Hyperparameter Tuning for Decision Tree\n",
        "param_grid_dt = {'max_depth': [3, 5, 7, 10, None], 'min_samples_split': [2, 5, 10]}\n",
        "grid_dt = GridSearchCV(DecisionTreeClassifier(random_state=42, class_weight='balanced'), param_grid_dt, cv=5, scoring='recall')\n",
        "grid_dt.fit(X_train, y_train)\n",
        "\n",
        "best_dt = grid_dt.best_estimator_\n",
        "print(\"Best params for Decision Tree:\", grid_dt.best_params_)\n",
        "\n",
        "# Evaluate tuned model\n",
        "y_pred_dt_tuned = best_dt.predict(X_test)\n",
        "recall_dt_tuned = recall_score(y_test, y_pred_dt_tuned)\n",
        "precision_dt_tuned = precision_score(y_test, y_pred_dt_tuned)\n",
        "print(f\"Tuned DT - Recall: {recall_dt_tuned:.3f}, Precision: {precision_dt_tuned:.3f}\")\n"
    ]
})

# Add hyperparameter tuning for Random Forest
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Hyperparameter Tuning for Random Forest\n",
        "param_grid_rf = {'n_estimators': [100, 200, 300], 'max_depth': [10, 20, None], 'min_samples_split': [2, 5]}\n",
        "grid_rf = GridSearchCV(RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1), param_grid_rf, cv=5, scoring='recall')\n",
        "grid_rf.fit(X_train, y_train)\n",
        "\n",
        "best_rf = grid_rf.best_estimator_\n",
        "print(\"Best params for Random Forest:\", grid_rf.best_params_)\n",
        "\n",
        "# Evaluate tuned model\n",
        "y_pred_rf_tuned = best_rf.predict(X_test)\n",
        "recall_rf_tuned = recall_score(y_test, y_pred_rf_tuned)\n",
        "precision_rf_tuned = precision_score(y_test, y_pred_rf_tuned)\n",
        "print(f\"Tuned RF - Recall: {recall_rf_tuned:.3f}, Precision: {precision_rf_tuned:.3f}\")\n"
    ]
})

# Add Learning Curves
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Learning Curves\n",
        "from sklearn.model_selection import learning_curve\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "models = [\n",
        "    (best_lr, 'Tuned Logistic Regression'),\n",
        "    (best_dt, 'Tuned Decision Tree'),\n",
        "    (best_rf, 'Tuned Random Forest')\n",
        "]\n",
        "\n",
        "fig, axes = plt.subplots(1, 3, figsize=(18, 5))\n",
        "\n",
        "for idx, (model, name) in enumerate(models):\n",
        "    train_sizes, train_scores, val_scores = learning_curve(\n",
        "        model, X_train, y_train, cv=5, scoring='recall', \n",
        "        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10)\n",
        "    )\n",
        "    train_mean = np.mean(train_scores, axis=1)\n",
        "    val_mean = np.mean(val_scores, axis=1)\n",
        "    train_std = np.std(train_scores, axis=1)\n",
        "    val_std = np.std(val_scores, axis=1)\n",
        "    \n",
        "    axes[idx].plot(train_sizes, train_mean, 'o-', label='Training score')\n",
        "    axes[idx].plot(train_sizes, val_mean, 'o-', label='Validation score')\n",
        "    axes[idx].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)\n",
        "    axes[idx].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)\n",
        "    axes[idx].set_xlabel('Training Size')\n",
        "    axes[idx].set_ylabel('Recall')\n",
        "    axes[idx].set_title(f'Learning Curve - {name}')\n",
        "    axes[idx].legend()\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()\n"
    ]
})

# Add Comparisons - ROC Curves
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Model Comparisons - ROC Curves\n",
        "from sklearn.metrics import roc_curve, auc\n",
        "\n",
        "models = [\n",
        "    (best_lr, y_pred_lr_tuned, 'Tuned Logistic Regression'),\n",
        "    (best_dt, y_pred_dt_tuned, 'Tuned Decision Tree'),\n",
        "    (best_rf, y_pred_rf_tuned, 'Tuned Random Forest')\n",
        "]\n",
        "\n",
        "plt.figure(figsize=(10, 8))\n",
        "for model, pred, name in models:\n",
        "    y_proba = model.predict_proba(X_test)[:, 1]\n",
        "    fpr, tpr, _ = roc_curve(y_test, y_proba)\n",
        "    roc_auc = auc(fpr, tpr)\n",
        "    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})')\n",
        "\n",
        "plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')\n",
        "plt.xlabel('False Positive Rate')\n",
        "plt.ylabel('True Positive Rate')\n",
        "plt.title('ROC Curves Comparison')\n",
        "plt.legend()\n",
        "plt.show()\n"
    ]
})

# Update Business Impact with tuned models
nb['cells'].append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Updated Business Impact Comparison with Tuned Models\n",
        "retention_cost = 50\n",
        "customer_ltv = 500\n",
        "\n",
        "models_business = []\n",
        "\n",
        "for name, pred in [('Tuned Logistic Regression', y_pred_lr_tuned), \n",
        "                   ('Tuned Decision Tree', y_pred_dt_tuned), \n",
        "                   ('Tuned Random Forest', y_pred_rf_tuned)]:\n",
        "    cm = confusion_matrix(y_test, pred)\n",
        "    tn, fp, fn, tp = cm.ravel()\n",
        "    \n",
        "    total_interventions = tp + fp\n",
        "    total_cost = total_interventions * retention_cost\n",
        "    value_saved = tp * customer_ltv\n",
        "    net_benefit = value_saved - total_cost\n",
        "    roi = (net_benefit / total_cost * 100) if total_cost > 0 else 0\n",
        "    \n",
        "    models_business.append({\n",
        "        'Model': name,\n",
        "        'Interventions': total_interventions,\n",
        "        'Churners Saved': tp,\n",
        "        'False Alarms': fp,\n",
        "        'Missed Churners': fn,\n",
        "        'Total Cost': f'${total_cost:,.0f}',\n",
        "        'Value Saved': f'${value_saved:,.0f}',\n",
        "        'Net Benefit': f'${net_benefit:,.0f}',\n",
        "        'ROI': f'{roi:.1f}%'\n",
        "    })\n",
        "\n",
        "business_comparison = pd.DataFrame(models_business)\n",
        "print(\"\\n\" + \"=\" * 80)\n",
        "print(\"BUSINESS IMPACT COMPARISON - TUNED MODELS\")\n",
        "print(\"=\" * 80)\n",
        "print(business_comparison.to_string(index=False))\n",
        "print(\"=\" * 80)\n"
    ]
})

with open('notebooks/eda.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)