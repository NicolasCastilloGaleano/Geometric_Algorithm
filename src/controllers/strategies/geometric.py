from src.constants.error import ERROR_INCOMPATIBLE_SIZES
from src.models.core.system import System
from src.constants.base import NET_LABEL, STR_ZERO
from src.funcs.base import ABECEDARY
from src.middlewares.slogger import SafeLogger
from src.funcs.base import emd_efecto
from src.models.base.sia import SIA
from src.constants.base import (
    ACTUAL,
    EFECTO,
    TYPE_TAG,
)
from src.constants.models import (
    GEOMETRIC_ANALYSIS_TAG,
    GEOMETRIC_LABEL,
    GEOMETRIC_STRAREGY_TAG,
)
from src.controllers.manager import Manager
from src.funcs.format import fmt_biparte_q
from src.middlewares.profile import profiler_manager, profile
from src.models.core.solution import Solution
import numpy as np
import time
from typing import List, Dict, Tuple

from concurrent.futures import ThreadPoolExecutor
import itertools

class GeometricSIA(SIA):
    def __init__(self, gestor: Manager):
        super().__init__(gestor)
        profiler_manager.start_session(
            f"{NET_LABEL}{len(gestor.estado_inicial)}{gestor.pagina}"
        )
        self.etiquetas = [tuple(s.lower() for s in ABECEDARY), ABECEDARY]
        self.logger = SafeLogger(GEOMETRIC_STRAREGY_TAG)
        self.tabla_transiciones: dict ={}
        self.vertices :set[tuple]
        self.tabla :dict[int, list[tuple[int, int]]] = {}
        self.memoria_particiones: dict[tuple[int, int], tuple[float, float]] = {}

    @profile(context={TYPE_TAG: GEOMETRIC_ANALYSIS_TAG})
    def aplicar_estrategia(
        self,
        condicion: str,
        alcance: str,
        mecanismo: str,
    ):
        """ vamos a hacer que vaya desde el estado inicial hasta el final, bit a bit diferente, llenando la tabla primero para distancias hamming 1 hasta n, con n la cantidad de bits que cambian del estado inicial al final. para esto podemos usar una tabla de transiciones, donde cada fila es un estado y cada columna es un bit. la tabla de transiciones se llena con los estados que se pueden alcanzar desde el estado inicial, y luego se va llenando la tabla de distancias hamming. para esto vamos a usar una lista de listas, donde cada lista es una fila de la tabla de transiciones. la primera fila es el estado inicial, y las siguientes filas son los estados alcanzables desde el estado inicial. la última fila es el estado final.
        paso a paso
        1. cargar la matriz, pasar a ncubos
        2. condicionar
        3. obtener los bits que cambian entre el estado inicial y el final
        4. obener vecinos del estado final que van hacia el estado inicial y calcular el costo de la transicion.
        5. para cada vecino, obtener los vecinos que van hacia el estado inicial y calcular el costo de la transicion.
        6. repetir hasta llegar al estado inicial.


        nota: intentar llenar la tabla desde el estado final hacia atras, pues al contrario habra dependencia de los valores de la tabla de los estados que van en camino hacia el estado final
        """
        # 1. Construir la representación n-dimensional del sistema
        self.sia_preparar_subsistema(condicion, alcance, mecanismo)
        futuro = tuple(
            (EFECTO, efecto) for efecto in self.sia_subsistema.indices_ncubos
        )
        presente = tuple(
            (ACTUAL, actual) for actual in self.sia_subsistema.dims_ncubos
        )  #
        self.vertices = set(presente + futuro)
        dims = self.sia_subsistema.dims_ncubos
        self.estado_inicial = self.sia_subsistema.estado_inicial[dims]
        self.estado_final = 1 - self.estado_inicial
        mip = self.find_mip()
        fmt_mip = fmt_biparte_q(list(mip), self.nodes_complement(mip))

        return Solution(
            estrategia= GEOMETRIC_LABEL,
            perdida=self.memoria_particiones[mip][0],
            distribucion_subsistema=self.sia_dists_marginales,
            distribucion_particion=self.memoria_particiones[mip][1],
            tiempo_total=time.time() - self.sia_tiempo_inicio,
            particion=fmt_mip,
        )
    
    def nodes_complement(self, nodes: list[tuple[int, int]]):
        return list(set(self.vertices) - set(nodes))
    
    def find_mip(self):
        """
        Implementa el algoritmo para encontrar la bipartición óptima
        utilizando el enfoque geométrico-topológico.
        """
        # 2. Calcular la tabla de costos T para cada variable
        self.sia_logger.critic("empieza.")
        # dims tendra los indices a tener en cuenta del estado inicial pues las otras dimensiones fueron marginalizadas
        estado_inicial = self.estado_inicial
        #el estado final sera el complementario del estado inicial
        estado_final = self.estado_final
        # print(f"Estado Inicial: {estado_inicial}, Estado Final: {estado_final}")
        self.idx_ncubos = list(range(len(self.sia_subsistema.indices_ncubos)))
        self.caminos: Dict[int, List[List[int]]] = {0: [estado_inicial.tolist()]}
        self.tabla_transiciones[tuple(self.caminos[0][0]),tuple(self.caminos[0][0])] = [0.0 for _ in range(len(self.sia_subsistema.indices_ncubos))]
        # self.tabla_transiciones[tuple(self.caminos[0][0]),tuple(estado_final)] = self.calcular_costo_temporal(
        #     self.caminos[0][0], estado_final, self.idx_ncubos)
        for nivel in range(1, len(estado_inicial)+1):
            # self.generar_caminos_hacia_complemento(estado_final,nivel)
            # for estado in self.caminos[nivel]:
            #     self.calcular_costo(self.caminos[0][0],estado,list(range(len(self.sia_subsistema.indices_ncubos))))
            self.calcular_costos_nivel(estado_final,nivel)
            # self.analizar_nivel(nivel)
            # print(f"Nivel {nivel}: {len(self.caminos[nivel])}")
            # añadir el promedio, quitar esto despues
            # self.tabla_transiciones[f"promedio nivel {nivel}                        "] = list(np.round(np.mean([self.tabla_transiciones[tuple(self.caminos[0][0]), tuple(estado)] for estado in self.caminos[nivel]], axis=0),decimals=4))
            # print(f"Promedio Nivel {nivel}: {promedio_nivel}")
            #analizar el nivel, por ejemplo, ver si en el primer nivel hay alguna particion sin costo para ninguna variable furura, si esto se da, podemos tomar esta particion. ademas, ver cual o cuales variables futuras tienen menos costo con al cambio de las variables presentes, si alguna llega a mostrar un comportamiento sin costo para todas las variables presentes, podemos pensar que esta posiblemente generara una particion optima en los niveles superiores.
        # 3. Identificar las biparticiones candidatas
        candidatos = self.identificar_particiones_optimas()
        # 4. Evaluar y seleccionar la bipartición óptima
        # print("Candidatos:")
        for idx, (presentes, futuros) in enumerate(candidatos):
            presentes = self.sia_subsistema.dims_ncubos[presentes]
            futuros = self.sia_subsistema.indices_ncubos[futuros]
            # print(f"Candidato {idx+1}: Presentes: {presentes}, Futuros: {futuros}")
            dist =self.sia_subsistema.bipartir(futuros,presentes).distribucion_marginal()
            emd = emd_efecto(dist, self.sia_dists_marginales)
            # print(f"EMD: {emd}, Distancia: {dist}")
            key = [(0,nodo) for nodo in presentes]
            key.extend([(1,nodo) for nodo in futuros])
            # print(key)
            self.memoria_particiones[tuple(key)] = (emd, dist)
        # 5. Retornar el resultado en formato compatible
        # for key, value in self.tabla_transiciones.items():
        #     print(f"Estado: {key}, Costo: {value}")
        # self.sia_logger.critic("termina")
        return min(
            self.memoria_particiones, key=lambda k: self.memoria_particiones[k][0]
        )

    def calcular_costo(self, estado_inicial:tuple, estado_final:tuple, ncubos:list[int]):
        """
            Funcion encargada de calcular el costo de transicion de transicion del estado inicial al estado final
            para las variables futuras definidas en ncubos
            aplica la funcion de costo tx(i,j)= y(|X[i]-X[j]|+ sum(tx(k,j)))
            donde:
                - y es el factor de decrecimiento 1/2^(dh(i,j))
                - dh(i,j) es la distancia hamming entre i y j
                - X[i] es el valor de probabilida de transicion de un estado para cada variable futura
                - sum(tx(i,k)) son todos costos de transicion de los vecinos de j que estan en un 
                  camino optimo desde i
        """
        key = tuple(estado_inicial), tuple(estado_final)
        # print(f"Calculando costo de transicion de {key}")
        if key not in self.tabla_transiciones:
            self.tabla_transiciones[key] = [None]*len(self.sia_subsistema.indices_ncubos)
            #quitar esto luego
            # self.tabla_transiciones[str(key)+"__"] = [None]*len(self.sia_subsistema.indices_ncubos)
        distancia_hamming = self.hamming(estado_inicial, estado_final)
        factor = 1/(2**distancia_hamming)
        index_inicial = tuple(np.array(estado_inicial)[::-1])
        index_final = tuple(np.array(estado_final)[::-1])
        for idx in ncubos:
            #se se aplica el factor aqui, este no afectara a los valores sumados de los vecinos
            #no genera los mismos resultados de la documentacion
            self.tabla_transiciones[key][idx] = (abs(self.sia_subsistema.ncubos[idx].data[index_inicial]-self.sia_subsistema.ncubos[idx].data[index_final]))
        #voy a revisar cuanto afectan los valores sumados en los siguiemtes pasos al valor que va guardado hasta este punto
        #este tmp1 luego lo quito, es solo para ver como afecta el factor a los valores de la tabla
        # tmp1 = self.tabla_transiciones[key].copy()
        if distancia_hamming > 1:
            for i in range(len(estado_inicial)):
                if estado_inicial[i] != estado_final[i]:
                    nuevo_estado = estado_final.copy()
                    # nuevo_estado = estado_inicial.copy()
                    nuevo_estado[i] = estado_inicial[i]
                    # nuevo_estado[i] = estado_final[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    temp_key = tuple(estado_inicial), nuevo_estado_tuple
                    # temp_key = nuevo_estado_tuple,tuple(estado_final)
                    # if temp_key not in self.tabla_transiciones:
                    #     print(f"Calculando de {temp_key}")
                    #     self.calcular_costo(nuevo_estado, estado_final, ncubos)
                    for n in ncubos:
                        self.tabla_transiciones[key][n] = self.tabla_transiciones[key][n] + self.tabla_transiciones[temp_key][n]
        # lo que sigue es aplicar el factor a todas las operaciones incluidas las que se sumaron de los vecinos
        # esta genera los resultados de la tabla de la documentacion
        tmp =[]
        for i,n in enumerate(self.tabla_transiciones[key]):
            if n is not None:
                tmp.append(factor * n)
                # aqui voy a terminar de revisar la diferencia entre tmp1 y el valor con las sumas, quitar luego
                # self.tabla_transiciones[str(key)+"__"][i] = round(tmp[-1] - (factor*tmp1[i]),4)
            else:
                tmp.append(n)
        self.tabla_transiciones[key] = tmp

    # def calcular_costo_temporal(self, estado_inicial:tuple, estado_final:tuple, ncubos:list[int], temp : dict = {}):
    #     """
    #         Funcion encargada de calcular el costo de transicion de transicion del estado inicial al estado final
    #         para las variables futuras definidas en ncubos
    #         aplica la funcion de costo tx(i,j)= y(|X[i]-X[j]|+ sum(tx(k,j)))
    #         donde:
    #             - y es el factor de decrecimiento 1/2^(dh(i,j))
    #             - dh(i,j) es la distancia hamming entre i y j
    #             - X[i] es el valor de probabilida de transicion de un estado para cada variable futura
    #             - sum(tx(i,k)) son todos costos de transicion de los vecinos de j que estan en un 
    #               camino optimo desde i
    #     """
    #     key = tuple(estado_inicial), tuple(estado_final)
    #     if key in temp:
    #         return temp[key]
    #     temp[key] = [None]*len(self.sia_subsistema.indices_ncubos)
    #     distancia_hamming = self.hamming(estado_inicial, estado_final)
    #     factor = 1/(2**distancia_hamming)
    #     index_inicial = tuple(np.array(estado_inicial)[::-1])
    #     index_final = tuple(np.array(estado_final)[::-1])
    #     for idx in ncubos:
    #         #se se aplica el factor aqui, este no afectara a los valores sumados de los vecinos
    #         #no genera los mismos resultados de la documentacion
    #         temp[key][idx] = (abs(self.sia_subsistema.ncubos[idx].data[index_inicial]-self.sia_subsistema.ncubos[idx].data[index_final]))
    #     #voy a revisar cuanto afectan los valores sumados en los siguiemtes pasos al valor que va guardado hasta este punto
    #     #este tmp1 luego lo quito, es solo para ver como afecta el factor a los valores de la tabla
    #     # tmp1 = self.tabla_transiciones[key].copy()
    #     if distancia_hamming > 1:
    #         for i in range(len(estado_inicial)):
    #             if estado_inicial[i] != estado_final[i]:
    #                 # nuevo_estado = estado_final.copy()
    #                 nuevo_estado = estado_inicial.copy()
    #                 # nuevo_estado[i] = estado_inicial[i]
    #                 nuevo_estado[i] = estado_final[i]

    #                 # veci = self.calcular_costo_temporal(nuevo_estado, estado_final, ncubos, temp)
    #                 for n in ncubos:
    #                     temp[key][n] = temp[key][n] + veci[n]
    #     # lo que sigue es aplicar el factor a todas las operaciones incluidas las que se sumaron de los vecinos
    #     # esta genera los resultados de la tabla de la documentacion
    #     tmp =[]
    #     for i,n in enumerate(temp[key]):
    #         if n is not None:
    #             tmp.append(factor * n)
    #             # aqui voy a terminar de revisar la diferencia entre tmp1 y el valor con las sumas, quitar luego
    #             # self.tabla_transiciones[str(key)+"__"][i] = round(tmp[-1] - (factor*tmp1[i]),4)
    #         else:
    #             tmp.append(n)
    #     temp[key] = tmp    
    #     return temp[key]
    
    def calcular_costos_nivel(self,estado_final: np.ndarray, nivel):
        n = len(estado_final)      
        visitados:set[tuple] = set()
        self.caminos[nivel] = []
        # if complementar:
        #     self.caminos[len(estado_final)-nivel] = []
        for estado_anterior in self.caminos[nivel - 1]:
            estado_actual = np.array(estado_anterior)
            for i in range(n):
                if estado_actual[i] != estado_final[i]:
                    nuevo_estado = estado_actual.copy()
                    nuevo_estado[i] = estado_final[i]
                    nuevo_estado_tuple = tuple(nuevo_estado)
                    if nuevo_estado_tuple not in visitados:
                        self.caminos[nivel].append(nuevo_estado.tolist())
                        self.calcular_costo(self.caminos[0][0],nuevo_estado.tolist(),self.idx_ncubos)
                        # if complementar:
                        #     # Calcular el estado complementario
                        #     estado_complementario = 1 - nuevo_estado
                        #     self.caminos[len(estado_final)-nivel].append(estado_complementario.tolist())
                        #     self.calcular_costo(self.caminos[0][0],estado_complementario.tolist(),self.idx_ncubos)
                        visitados.add(nuevo_estado_tuple)
    
    def identificar_particiones_optimas(self):
        """
        Identifica las particiones óptimas basadas en los costos de transición
        y las distancias Hamming entre los estados.
        """
        # Implementar la lógica para identificar particiones óptimas
        # basadas en la tabla de transiciones y los costos calculados.
        # en el nivel cero solo se busca en el ultimo nivel que variable futura genera el menor costo de transicion
        idx_nivel_cero = 0
        costo=1e5
        key = tuple(self.caminos[0][0]), tuple(self.estado_final)
        # print(key)
        costos: list = self.tabla_transiciones[key]
        # print(costos)
        for idx, valor in enumerate(costos):
            if valor < costo:
                # print(valor)
                costo = valor
                idx_nivel_cero = idx
        presentes_nivel_cero = [i for i in range(len(self.estado_final))]
        furutros_nivel_cero = [i for i in range(len(self.sia_subsistema.indices_ncubos)) if i != idx_nivel_cero]
        candidatos = [[presentes_nivel_cero, furutros_nivel_cero]]
        #reccorrer la tabla solo hasta la mitad, pues las particiones son simetricas
        #si la cantidad de niveles es par, se debe considerar el nivel central como un caso especial pues se añadira el nivel cero
        es_par = len(self.caminos) % 2 == 0
        if es_par:
            mitad = len(self.caminos) // 2
        else:
            mitad = (len(self.caminos) // 2) +1
        # print(mitad)
        for nivel in range(1,mitad):
            candidato_nivel = self.caminos[nivel][0]
            costo_candidato_nivel = 1e5
            presentes_nivel = []
            futuros_nivel = []
            for estado in self.caminos[nivel]:
                candidato = estado
                costo_candidato = 0
                presentes = []
                futuros = []
                # revisar si el estado es una particion optima
                actual = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado)), None)
                estado_complementario = (1-np.array(estado)).tolist()
                complementario = self.tabla_transiciones.get((tuple(self.caminos[0][0]), tuple(estado_complementario)), None)
                # print(f"{nivel}, E: {estado}, E': {estado_complementario}, E-> {actual}, E' -> {complementario}")
                for idx,i in enumerate(estado):
                    if i == self.caminos[0][0][idx]:
                        presentes.append(idx)
                for idx,_ in enumerate(self.idx_ncubos):
                    if actual[idx] <= complementario[idx]:
                        futuros.append(idx)
                        costo_candidato += actual[idx]
                    else:
                        costo_candidato += complementario[idx]
                if costo_candidato < costo_candidato_nivel:
                    candidato_nivel = candidato
                    costo_candidato_nivel = costo_candidato
                    presentes_nivel = presentes
                    futuros_nivel = futuros
            candidatos.append([presentes_nivel, futuros_nivel])
        return candidatos

    def hamming(self,a: List[int], b: List[int]) -> int:
        return sum(x != y for x, y in zip(a, b))