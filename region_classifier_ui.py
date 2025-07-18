import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
from region_classifier import SmallSampleRegionClassifier
import threading

class RegionClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Toddalia asiatica （L.） Lam. Region Classifier")
        self.root.geometry("800x600")

        self.classifier = None
        self.train_data = None

        self.setup_ui()

    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        ttk.Label(main_frame, text="Toddalia asiatica （L.） Lam. Region Classifier", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=3, pady=10)

        ttk.Button(main_frame, text="Load Training Data", command=self.load_training_data).grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)
        self.train_status = ttk.Label(main_frame, text="No training data loaded")
        self.train_status.grid(row=1, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Button(main_frame, text="Train Model", command=self.train_model).grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.model_status = ttk.Label(main_frame, text="Model not trained")
        self.model_status.grid(row=2, column=1, padx=5, pady=5, sticky=tk.W)

        separator = ttk.Separator(main_frame, orient='horizontal')
        separator.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        ttk.Label(main_frame, text="Predict New Samples:", font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=3, pady=5, sticky=tk.W)

        ttk.Button(main_frame, text="Select Excel File", command=self.load_prediction_data).grid(row=5, column=0, padx=5, pady=5, sticky=tk.W)
        self.file_label = ttk.Label(main_frame, text="No file selected")
        self.file_label.grid(row=5, column=1, padx=5, pady=5, sticky=tk.W)

        ttk.Label(main_frame, text="Or manually input 40 component values (comma separated):").grid(row=6, column=0, columnspan=3, pady=5, sticky=tk.W)

        self.manual_input = tk.Text(main_frame, height=3, width=80)
        self.manual_input.grid(row=7, column=0, columnspan=3, padx=5, pady=5, sticky=(tk.W, tk.E))

        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=8, column=0, columnspan=3, padx=5, pady=10, sticky=tk.W)

        ttk.Button(button_frame, text="Start Prediction", command=self.predict).grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="Explain Prediction", command=self.explain_prediction).grid(row=0, column=1, padx=5)
        ttk.Button(button_frame, text="Save Model", command=self.save_model).grid(row=0, column=2, padx=5)
        ttk.Button(button_frame, text="Load Model", command=self.load_model).grid(row=0, column=3, padx=5)

        self.result_frame = ttk.LabelFrame(main_frame, text="Prediction Results", padding="10")
        self.result_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)

        self.result_tree = ttk.Treeview(self.result_frame, columns=('sample', 'prediction', 'confidence'), show='headings', height=10)
        self.result_tree.heading('sample', text='Sample')
        self.result_tree.heading('prediction', text='Predicted Region')
        self.result_tree.heading('confidence', text='Confidence')
        self.result_tree.column('sample', width=100)
        self.result_tree.column('prediction', width=200)
        self.result_tree.column('confidence', width=100)

        scrollbar = ttk.Scrollbar(self.result_frame, orient=tk.VERTICAL, command=self.result_tree.yview)
        self.result_tree.configure(yscrollcommand=scrollbar.set)

        self.result_tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(9, weight=1)
        self.result_frame.columnconfigure(0, weight=1)
        self.result_frame.rowconfigure(0, weight=1)

    def load_training_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Training Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            try:
                train_data = pd.read_excel(file_path, sheet_name=0, header=None)

                # 正确提取特征数据（从第3行开始，第2列到第41列，共40个特征）
                X_train = train_data.iloc[2:, 1:41].values.astype(float)

                # 正确提取地区标签（第1列），并去掉样本编号
                raw_labels = train_data.iloc[2:, 0].values
                y_train = []
                for label in raw_labels:
                    # 去掉最后的样本编号 (如 Y-FJ-1 -> Y-FJ)
                    parts = label.split('-')
                    if len(parts) >= 3:
                        region = '-'.join(parts[:-1])
                    else:
                        region = label
                    y_train.append(region)

                self.train_data = {
                    'X': X_train,
                    'y': np.array(y_train),
                    'file_path': file_path
                }

                # 显示正确的统计信息
                unique_regions = len(np.unique(y_train))
                self.train_status.config(text=f"Loaded: {len(X_train)} samples, {X_train.shape[1]} features, {unique_regions} regions")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load training data: {str(e)}")

    def train_model(self):
        if self.train_data is None:
            messagebox.showwarning("Warning", "Please load training data first")
            return

        def train_thread():
            try:
                self.progress.start()
                self.classifier = SmallSampleRegionClassifier(enable_logging=True)
                self.classifier.train(self.train_data['X'], self.train_data['y'])
                self.progress.stop()
                self.model_status.config(text="Model training completed")
                messagebox.showinfo("Success", "Model training completed")
            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Model training failed: {str(e)}")

        threading.Thread(target=train_thread, daemon=True).start()

    def load_prediction_data(self):
        file_path = filedialog.askopenfilename(
            title="Select Prediction Data File",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if file_path:
            self.prediction_file = file_path
            self.file_label.config(text=f"Selected: {file_path.split('/')[-1]}")

    def predict(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        def predict_thread():
            try:
                self.progress.start()

                for item in self.result_tree.get_children():
                    self.result_tree.delete(item)

                manual_text = self.manual_input.get("1.0", tk.END).strip()

                if manual_text:
                    values = [float(x.strip()) for x in manual_text.split(',')]
                    if len(values) != 40:
                        raise ValueError(f"Need 40 values, but got {len(values)}")
                    X_pred = np.array([values])
                    sample_names = ["Manual Input"]

                elif hasattr(self, 'prediction_file'):
                    pred_data = pd.read_excel(self.prediction_file, header=None)
                    X_pred = []
                    sample_names = []

                    # 检查数据格式：如果第3行开始有数据，说明是新格式（行为样本）
                    if pred_data.shape[0] > 2:
                        # 新格式：每行是一个样本，从第3行开始
                        for row_idx in range(2, pred_data.shape[0]):
                            try:
                                # 提取特征数据（第2列到第41列，共40个特征）
                                sample = pd.to_numeric(pred_data.iloc[row_idx, 2:42], errors='coerce').values
                                if not np.isnan(sample).all() and len(sample) == 40:
                                    X_pred.append(sample)
                                    # 使用样本ID作为名称
                                    sample_id = pred_data.iloc[row_idx, 0]
                                    region_label = pred_data.iloc[row_idx, 1]
                                    sample_names.append(f"Sample_{sample_id}_{region_label}")
                            except:
                                continue
                    else:
                        # 旧格式：每列是一个样本
                        for col_idx in range(1, pred_data.shape[1]):
                            col_name = pred_data.columns[col_idx] if pred_data.columns[col_idx] else f"Sample_{col_idx}"
                            try:
                                sample = pd.to_numeric(pred_data.iloc[:40, col_idx], errors='coerce').values
                                if not np.isnan(sample).all() and len(sample) == 40:
                                    X_pred.append(sample)
                                    sample_names.append(col_name)
                            except:
                                continue

                    if not X_pred:
                        raise ValueError("No valid prediction data found")
                    X_pred = np.array(X_pred)
                else:
                    raise ValueError("Please select a file or input data manually")

                predictions = self.classifier.predict(X_pred)
                probabilities = self.classifier.predict_proba(X_pred)

                # 保存预测结果用于解释
                self.last_predictions = predictions
                self.last_X_pred = X_pred

                for i, (sample_name, pred, prob) in enumerate(zip(sample_names, predictions, probabilities)):
                    confidence = prob.max()
                    self.result_tree.insert('', 'end', values=(sample_name, pred, f"{confidence:.3f}"))

                self.progress.stop()

            except Exception as e:
                self.progress.stop()
                messagebox.showerror("Error", f"Prediction failed: {str(e)}")

        threading.Thread(target=predict_thread, daemon=True).start()

    def explain_prediction(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "Please train the model first")
            return

        if not hasattr(self, 'last_predictions') or self.last_predictions is None:
            messagebox.showwarning("Warning", "Please make a prediction first")
            return

        try:
            explanation = self.classifier.explain_prediction(self.last_X_pred, 0)

            explain_window = tk.Toplevel(self.root)
            explain_window.title("Prediction Explanation")
            explain_window.geometry("500x400")

            text_widget = tk.Text(explain_window, wrap=tk.WORD, padx=10, pady=10)
            text_widget.pack(fill=tk.BOTH, expand=True)

            explanation_text = f"""Prediction Explanation for First Sample:

Distance Metrics:
• Euclidean Distance: {explanation['euclidean_distance']:.4f}
• Manhattan Distance: {explanation['manhattan_distance']:.4f}
• Cosine Distance: {explanation['cosine_distance']:.4f}

Nearest Regions:
• Euclidean: {explanation['nearest_region_euclidean']}
• Manhattan: {explanation['nearest_region_manhattan']}
• Cosine: {explanation['nearest_region_cosine']}

Distance Weights:
• Euclidean: {explanation['distance_weights']['euclidean']:.3f}
• Manhattan: {explanation['distance_weights']['manhattan']:.3f}
• Cosine: {explanation['distance_weights']['cosine']:.3f}

The prediction is based on the weighted combination of these distance metrics.
The region with the smallest weighted distance is selected as the final prediction.
"""

            text_widget.insert(tk.END, explanation_text)
            text_widget.config(state=tk.DISABLED)

        except Exception as e:
            messagebox.showerror("Error", f"Failed to explain prediction: {str(e)}")

    def save_model(self):
        if self.classifier is None:
            messagebox.showwarning("Warning", "No model to save")
            return

        file_path = filedialog.asksaveasfilename(
            title="Save Model",
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            try:
                import pickle
                with open(file_path, 'wb') as f:
                    pickle.dump(self.classifier, f)
                messagebox.showinfo("Success", "Model saved successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save model: {str(e)}")

    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Load Model",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        if file_path:
            try:
                import pickle
                with open(file_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                self.model_status.config(text="Model loaded successfully")
                messagebox.showinfo("Success", "Model loaded successfully")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = RegionClassifierUI(root)
    root.mainloop()
