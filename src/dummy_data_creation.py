# -*- coding: utf-8 -*-
"""
Creación de Datos Dummy con Distribución Normal Multivariada
Ejercicio con NumPy y visualización con Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt

def problema_1_crear_randoms():
    """
    Problema 1: Crear 500 números aleatorios con distribución normal multivariada
    Media: (-3, 0), Matriz de covarianza: [[1.0, 0.8], [0.8, 1.0]]
    """
    print("=" * 70)
    print("PROBLEMA 1: CREACIÓN DE DATOS DUMMY")
    print("=" * 70)
    
    # Fijar semilla para reproducibilidad
    np.random.seed(0)
    
    # Definir parámetros de la distribución
    media = [-3, 0]
    matriz_covarianza = [[1.0, 0.8], [0.8, 1.0]]
    
    # Generar 500 muestras
    datos_1 = np.random.multivariate_normal(media, matriz_covarianza, 500)
    
    print(f"✓ Datos generados: {datos_1.shape}")
    print(f"✓ Media objetivo: {media}")
    print(f"✓ Matriz de covarianza:\n{np.array(matriz_covarianza)}")
    print(f"✓ Media real calculada: {np.mean(datos_1, axis=0).round(2)}")
    print(f"✓ Covarianza real calculada:\n{np.cov(datos_1.T).round(2)}")
    
    return datos_1

def problema_2_visualizar_scatter(datos_1):
    """
    Problema 2: Visualizar los datos con scatter plot
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 2: SCATTER PLOT")
    print("=" * 70)
    
    # Configuración para CodeSpaces
    plt.switch_backend('Agg')
    
    plt.figure(figsize=(10, 8))
    plt.scatter(datos_1[:, 0], datos_1[:, 1], alpha=0.6, color='blue', label='Cluster 0')
    plt.xlabel('Dimensión X')
    plt.ylabel('Dimensión Y')
    plt.title('Scatter Plot - Datos Cluster 0', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    plt.savefig('scatter_cluster_0.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Scatter plot guardado como 'scatter_cluster_0.png'")

def problema_3_visualizar_histogramas(datos_1):
    """
    Problema 3: Visualizar histogramas para cada dimensión
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 3: HISTOGRAMAS POR DIMENSIÓN")
    print("=" * 70)
    
    plt.figure(figsize=(12, 5))
    
    # Histograma para dimensión X
    plt.subplot(1, 2, 1)
    plt.hist(datos_1[:, 0], bins=20, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel('Valores Dimensión X')
    plt.ylabel('Frecuencia')
    plt.title('Histograma - Dimensión X', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(-6, 1)  # Mismo rango para ambos histogramas
    
    # Histograma para dimensión Y
    plt.subplot(1, 2, 2)
    plt.hist(datos_1[:, 1], bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.xlabel('Valores Dimensión Y')
    plt.ylabel('Frecuencia')
    plt.title('Histograma - Dimensión Y', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.xlim(-6, 1)  # Mismo rango para ambos histogramas
    
    plt.tight_layout()
    plt.savefig('histogramas_cluster_0.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Histogramas guardados como 'histogramas_cluster_0.png'")

def problema_4_crear_segundo_cluster(datos_1):
    """
    Problema 4: Crear segundo cluster y visualizar ambos
    Media: (0, -3), Matriz de covarianza: [[1.0, 0.8], [0.8, 1.0]]
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 4: SEGUNDO CLUSTER Y VISUALIZACIÓN CONJUNTA")
    print("=" * 70)
    
    np.random.seed(0)  # Misma semilla para reproducibilidad
    
    # Definir parámetros del segundo cluster
    media_2 = [0, -3]
    matriz_covarianza_2 = [[1.0, 0.8], [0.8, 1.0]]
    
    # Generar 500 muestras del segundo cluster
    datos_2 = np.random.multivariate_normal(media_2, matriz_covarianza_2, 500)
    
    print(f"✓ Segundo cluster generado: {datos_2.shape}")
    print(f"✓ Media objetivo cluster 2: {media_2}")
    print(f"✓ Media real cluster 2: {np.mean(datos_2, axis=0).round(2)}")
    
    # Visualizar ambos clusters juntos
    plt.figure(figsize=(10, 8))
    plt.scatter(datos_1[:, 0], datos_1[:, 1], alpha=0.6, color='blue', label='Cluster 0')
    plt.scatter(datos_2[:, 0], datos_2[:, 1], alpha=0.6, color='red', label='Cluster 1')
    plt.xlabel('Dimensión X')
    plt.ylabel('Dimensión Y')
    plt.title('Scatter Plot - Dos Clusters', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    plt.savefig('scatter_dos_clusters.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Scatter plot con dos clusters guardado como 'scatter_dos_clusters.png'")
    
    return datos_2

def problema_5_combinar_datos(datos_1, datos_2):
    """
    Problema 5: Combinar ambos datasets en uno solo
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 5: COMBINACIÓN DE DATOS")
    print("=" * 70)
    
    # Combinar usando np.concatenate
    datos_combinados = np.concatenate([datos_1, datos_2], axis=0)
    
    # Alternativa con np.vstack
    # datos_combinados = np.vstack([datos_1, datos_2])
    
    print(f"✓ Datos cluster 1 shape: {datos_1.shape}")
    print(f"✓ Datos cluster 2 shape: {datos_2.shape}")
    print(f"✓ Datos combinados shape: {datos_combinados.shape}")
    print(f"✓ Primeras 3 filas combinadas:\n{datos_combinados[:3]}")
    print(f"✓ Últimas 3 filas combinadas:\n{datos_combinados[-3:]}")
    
    return datos_combinados

def problema_6_agregar_etiquetas(datos_combinados, n_muestras_por_cluster=500):
    """
    Problema 6: Agregar columna de etiquetas (0 para cluster 1, 1 para cluster 2)
    """
    print("\n" + "=" * 70)
    print("PROBLEMA 6: ETIQUETADO DE DATOS")
    print("=" * 70)
    
    # Crear array de etiquetas
    etiquetas_cluster_0 = np.zeros(n_muestras_por_cluster)  # 0 para primer cluster
    etiquetas_cluster_1 = np.ones(n_muestras_por_cluster)   # 1 para segundo cluster
    
    # Combinar etiquetas
    etiquetas_combinadas = np.concatenate([etiquetas_cluster_0, etiquetas_cluster_1])
    
    # Agregar etiquetas como tercera columna
    datos_etiquetados = np.column_stack([datos_combinados, etiquetas_combinadas])
    
    print(f"✓ Shape datos etiquetados: {datos_etiquetados.shape}")
    print(f"✓ Etiquetas únicas: {np.unique(datos_etiquetados[:, 2])}")
    print(f"✓ Conteo por etiqueta:")
    print(f"  - Etiqueta 0: {np.sum(datos_etiquetados[:, 2] == 0)} muestras")
    print(f"  - Etiqueta 1: {np.sum(datos_etiquetados[:, 2] == 1)} muestras")
    
    print(f"\n✓ Primeras 5 filas (con etiquetas):")
    for i in range(5):
        print(f"  [{datos_etiquetados[i, 0]:.2f}, {datos_etiquetados[i, 1]:.2f}, {int(datos_etiquetados[i, 2])}]")
    
    print(f"\n✓ Últimas 5 filas (con etiquetas):")
    for i in range(-5, 0):
        print(f"  [{datos_etiquetados[i, 0]:.2f}, {datos_etiquetados[i, 1]:.2f}, {int(datos_etiquetados[i, 2])}]")
    
    return datos_etiquetados

def visualizacion_final(datos_etiquetados):
    """
    Visualización final con todos los datos etiquetados
    """
    print("\n" + "=" * 70)
    print("VISUALIZACIÓN FINAL")
    print("=" * 70)
    
    # Separar datos por etiqueta
    datos_0 = datos_etiquetados[datos_etiquetados[:, 2] == 0]
    datos_1 = datos_etiquetados[datos_etiquetados[:, 2] == 1]
    
    plt.figure(figsize=(15, 5))
    
    # 1. Scatter plot final
    plt.subplot(1, 3, 1)
    plt.scatter(datos_0[:, 0], datos_0[:, 1], alpha=0.6, color='blue', label='Cluster 0 (Etiqueta 0)')
    plt.scatter(datos_1[:, 0], datos_1[:, 1], alpha=0.6, color='red', label='Cluster 1 (Etiqueta 1)')
    plt.xlabel('Dimensión X')
    plt.ylabel('Dimensión Y')
    plt.title('Datos Finales Etiquetados', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    # 2. Histograma de dimensión X por etiqueta
    plt.subplot(1, 3, 2)
    plt.hist(datos_0[:, 0], bins=20, alpha=0.7, color='blue', label='Cluster 0', density=True)
    plt.hist(datos_1[:, 0], bins=20, alpha=0.7, color='red', label='Cluster 1', density=True)
    plt.xlabel('Dimensión X')
    plt.ylabel('Densidad')
    plt.title('Distribución Dimensión X', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 3. Histograma de dimensión Y por etiqueta
    plt.subplot(1, 3, 3)
    plt.hist(datos_0[:, 1], bins=20, alpha=0.7, color='blue', label='Cluster 0', density=True)
    plt.hist(datos_1[:, 1], bins=20, alpha=0.7, color='red', label='Cluster 1', density=True)
    plt.xlabel('Dimensión Y')
    plt.ylabel('Densidad')
    plt.title('Distribución Dimensión Y', fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('visualizacion_final.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✓ Visualización final guardada como 'visualizacion_final.png'")

def resumen_estadisticas(datos_etiquetados):
    """
    Muestra un resumen estadístico de los datos generados
    """
    print("\n" + "=" * 70)
    print("RESUMEN ESTADÍSTICO")
    print("=" * 70)
    
    # Separar por etiquetas
    datos_0 = datos_etiquetados[datos_etiquetados[:, 2] == 0]
    datos_1 = datos_etiquetados[datos_etiquetados[:, 2] == 1]
    
    print("CLUSTER 0 (Etiqueta 0):")
    print(f"  • Media X: {np.mean(datos_0[:, 0]):.2f}")
    print(f"  • Media Y: {np.mean(datos_0[:, 1]):.2f}")
    print(f"  • Desviación estándar X: {np.std(datos_0[:, 0]):.2f}")
    print(f"  • Desviación estándar Y: {np.std(datos_0[:, 1]):.2f}")
    print(f"  • Correlación X-Y: {np.corrcoef(datos_0[:, 0], datos_0[:, 1])[0,1]:.2f}")
    
    print("\nCLUSTER 1 (Etiqueta 1):")
    print(f"  • Media X: {np.mean(datos_1[:, 0]):.2f}")
    print(f"  • Media Y: {np.mean(datos_1[:, 1]):.2f}")
    print(f"  • Desviación estándar X: {np.std(datos_1[:, 0]):.2f}")
    print(f"  • Desviación estándar Y: {np.std(datos_1[:, 1]):.2f}")
    print(f"  • Correlación X-Y: {np.corrcoef(datos_1[:, 0], datos_1[:, 1])[0,1]:.2f}")
    
    print(f"\nDATASET COMPLETO:")
    print(f"  • Total de muestras: {len(datos_etiquetados)}")
    print(f"  • Balance: {len(datos_0)} vs {len(datos_1)} (50%-50%)")
    print(f"  • Shape final: {datos_etiquetados.shape}")

def main():
    """
    Función principal que ejecuta todos los problemas
    """
    print("🎯 CREACIÓN DE DATOS DUMMY - DISTRIBUCIÓN NORMAL MULTIVARIADA")
    print("=" * 70)
    
    try:
        # Problema 1: Crear primer cluster
        datos_1 = problema_1_crear_randoms()
        
        # Problema 2: Scatter plot del primer cluster
        problema_2_visualizar_scatter(datos_1)
        
        # Problema 3: Histogramas del primer cluster
        problema_3_visualizar_histogramas(datos_1)
        
        # Problema 4: Crear segundo cluster y visualizar ambos
        datos_2 = problema_4_crear_segundo_cluster(datos_1)  # ¡CORREGIDO! Ahora recibe datos_1
        
        # Problema 5: Combinar datos
        datos_combinados = problema_5_combinar_datos(datos_1, datos_2)
        
        # Problema 6: Agregar etiquetas
        datos_etiquetados = problema_6_agregar_etiquetas(datos_combinados)
        
        # Visualización final
        visualizacion_final(datos_etiquetados)
        
        # Resumen estadístico
        resumen_estadisticas(datos_etiquetados)
        
        # Mensaje final
        print("\n" + "=" * 70)
        print("✅ EJERCICIO COMPLETADO EXITOSAMENTE")
        print("=" * 70)
        print("📁 ARCHIVOS GENERADOS:")
        print("   • scatter_cluster_0.png")
        print("   • histogramas_cluster_0.png") 
        print("   • scatter_dos_clusters.png")
        print("   • visualizacion_final.png")
        print("\n🎯 APLICACIONES EN MACHINE LEARNING:")
        print("   • Dataset listo para algoritmos de clasificación")
        print("   • Prueba de modelos de clustering")
        print("   • Validación de algoritmos de reducción dimensional")
        print("   • Ejemplo de datos sintéticos balanceados")
        
    except Exception as e:
        print(f"❌ Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

# Ejecutar el programa
if __name__ == "__main__":
    main()
