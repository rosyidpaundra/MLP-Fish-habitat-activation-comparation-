from qgis.PyQt.QtCore import QCoreApplication
from qgis.core import (QgsProcessing, QgsProcessingAlgorithm, 
                       QgsProcessingParameterVectorLayer, QgsProcessingParameterRasterLayer,
                       QgsProcessingParameterRasterDestination, QgsRasterFileWriter,
                       Qgis, QgsRasterBlock, QgsPointXY)
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.utils import shuffle
from sklearn.inspection import permutation_importance

class PrediksiIkanMLPTargetFinal(QgsProcessingAlgorithm):
    TANGKAPAN = 'TANGKAPAN'; SST = 'SST'; CHL = 'CHL'; BATHY = 'BATHY'
    CORAL = 'CORAL'; SAL = 'SAL'; SSE = 'SSE'; CURRENT = 'CURRENT'
    OUTPUT = 'OUTPUT'

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterVectorLayer(self.TANGKAPAN, 'Titik Tangkapan', [QgsProcessing.TypeVectorPoint]))
        self.addParameter(QgsProcessingParameterRasterLayer(self.SST, 'Raster SST'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.CHL, 'Raster Chlorophyll-a'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.BATHY, 'Raster Batimetri'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.CORAL, 'Raster Coral Reef'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.SAL, 'Raster Salinitas'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.SSE, 'Raster Sea Surface Elevation'))
        self.addParameter(QgsProcessingParameterRasterLayer(self.CURRENT, 'Raster Sea Current Velocity'))
        self.addParameter(QgsProcessingParameterRasterDestination(self.OUTPUT, 'Hasil Prediksi MLP (Anti-Overfitting)'))

    def processAlgorithm(self, parameters, context, feedback):
        np.random.seed(42) # Menjamin konsistensi
        
        rasters = {
            'sst': self.parameterAsRasterLayer(parameters, self.SST, context),
            'chl': self.parameterAsRasterLayer(parameters, self.CHL, context),
            'bathy': self.parameterAsRasterLayer(parameters, self.BATHY, context),
            'coral': self.parameterAsRasterLayer(parameters, self.CORAL, context),
            'sal': self.parameterAsRasterLayer(parameters, self.SAL, context),
            'sse': self.parameterAsRasterLayer(parameters, self.SSE, context),
            'current': self.parameterAsRasterLayer(parameters, self.CURRENT, context)
        }
        points_layer = self.parameterAsVectorLayer(parameters, self.TANGKAPAN, context)
        output_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        # 1. Ekstraksi Data
        raw_presence = []
        for feature in points_layer.getFeatures():
            geom = feature.geometry().asPoint()
            row = []
            valid = True
            for k in rasters:
                val, _ = rasters[k].dataProvider().sample(QgsPointXY(geom.x(), geom.y()), 1)
                if np.isnan(val) or val == -9999: valid = False; break
                row.append(val)
            if valid: raw_presence.append(row)

        df_pres = pd.DataFrame(raw_presence)
        Q1 = df_pres.quantile(0.25); Q3 = df_pres.quantile(0.75); IQR = Q3 - Q1
        df_clean = df_pres[~((df_pres < (Q1 - 1.5 * IQR)) | (df_pres > (Q3 + 1.5 * IQR))).any(axis=1)]
        
        # 2. Augmentasi Noise (Ditingkatkan ke 0.12 untuk memecah overfitting)
        pres_vals = df_clean.values.tolist()
        for _ in range(len(pres_vals)):
            idx = np.random.randint(0, len(pres_vals))
            noise = np.random.normal(0, 0.12, size=len(rasters))
            pres_vals.append((pres_vals[idx] + (pres_vals[idx] * noise)).tolist())

        # Pseudo-Absence dengan variansi lebih lebar
        abs_vals = [[df_clean[i].mean() + (df_clean[i].std() * 1.0 * (1 if np.random.rand() > 0.5 else -1)) 
                     for i in range(len(rasters))] for _ in range(len(pres_vals))]
        
        X = np.vstack([pres_vals, abs_vals]); y = np.hstack([np.ones(len(pres_vals)), np.zeros(len(abs_vals))])
        X, y = shuffle(X, y, random_state=42)

        # 3. Training MLP dengan Regularisasi Sangat Kuat
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        mlp = MLPClassifier(
            hidden_layer_sizes=(3,), # Neuron dikurangi agar tidak terlalu "cerdas" menghafal
            activation='tanh', 
            solver='lbfgs', 
            alpha=5.0, # Alpha tinggi untuk menekan overfitting
            max_iter=1500, 
            random_state=42
        )
        
        cv_scores = cross_val_score(mlp, X_scaled, y, cv=5)
        feedback.pushInfo(f"Rata-rata Akurasi Cross-Validation (5-Fold): {cv_scores.mean():.4f}")
        
        mlp.fit(X_scaled, y)

        y_pred = mlp.predict(X_scaled)
        feedback.pushInfo("======== HASIL EVALUASI MODEL (ANTI-OVERFITTING) ========")
        feedback.pushInfo(f"Akurasi Pelatihan: {accuracy_score(y, y_pred):.4f}")
        feedback.pushInfo(classification_report(y, y_pred))

        # ANALISIS PERMUTATION IMPORTANCE
        feedback.pushInfo("======== ANALISIS VARIABEL ========")
        result = permutation_importance(mlp, X_scaled, y, n_repeats=10, random_state=42)
        feature_names = ['SST', 'Chlorophyll-a', 'Batimetri', 'Coral Reef', 'Salinitas', 'SSE', 'Current Velocity']
        for i in result.importances_mean.argsort()[::-1]:
            feedback.pushInfo(f"{feature_names[i]}: {result.importances_mean[i]:.4f}")

        # 4. Prediksi Spasial
        base = rasters['sst']
        rows, cols = base.height(), base.width()
        writer = QgsRasterFileWriter(output_path)
        out_provider = writer.createOneBandRaster(Qgis.DataType.Float32, cols, rows, base.extent(), base.crs())
        out_provider.setNoDataValue(1, -9999.0)
        
        blocks = {k: rasters[k].dataProvider().block(1, base.extent(), cols, rows) for k in rasters}
        out_block = QgsRasterBlock(Qgis.DataType.Float32, cols, rows)

        for r in range(rows):
            if feedback.isCanceled(): break
            for c in range(cols):
                pix = []; valid = True
                for k in rasters:
                    v = blocks[k].value(r, c)
                    if np.isnan(v) or v == -9999: valid = False; break
                    pix.append(v)
                
                if valid:
                    prob = mlp.predict_proba(scaler.transform([pix]))[0][1]
                    out_block.setValue(r, c, float(prob))
                else:
                    out_block.setValue(r, c, -9999.0)
            feedback.setProgress(int((r/rows)*100))

        out_provider.writeBlock(out_block, 1, 0, 0)
        return {self.OUTPUT: output_path}

    def name(self): return 'mlp_anti_overfitting'
    def displayName(self): return 'MLP Fishing Ground Deep Learning by Rosyid Paundra'
    def group(self): return 'Perikanan'
    def groupId(self): return 'perikanan'
    def createInstance(self): return PrediksiIkanMLPTargetFinal()
