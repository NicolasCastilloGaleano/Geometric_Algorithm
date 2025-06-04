from src.controllers.manager import Manager

from src.controllers.strategies.force import BruteForce
from src.controllers.strategies.q_nodes import QNodes
from src.controllers.strategies.geometric import GeometricSIA


def iniciar():
    """Punto de entrada principal"""
                    # ABCD #
    # estado_inicial = "100"
    # condiciones =    "111"
    # alcance =        "111"
    # mecanismo =      "111"
    # estado_inicial = "0000"
    # condiciones =    "1111"
    # alcance =        "1111"
    # mecanismo =      "1111"
    # estado_inicial = "1000"
    # condiciones =    "1111"
    # alcance =        "0111"
    # mecanismo =      "1111"
    # estado_inicial = "100000"
    # condiciones =    "111111"
    # alcance =        "101011"
    # mecanismo =      "111111"
    # estado_inicial = "100000"
    # condiciones =    "111111"
    # alcance =        "111111"
    # mecanismo =      "111111"
    # estado_inicial = "100000"
    # condiciones =    "111111"
    # alcance =        "111111"
    # mecanismo =      "011111"
    # estado_inicial = "1000000000"
    # condiciones =    "1111111111"
    # alcance =        "1111111111"
    # mecanismo =      "1111111111"
    estado_inicial = "1000000000"
    condiciones =    "1111111111"
    alcance =        "0101010101"
    mecanismo =      "1111111111"
    # estado_inicial = "1000000000"
    # condiciones =    "1111111111"
    # alcance =        "1111111110"
    # mecanismo =      "1111111111"
    # estado_inicial = "10000000000000000000"
    # condiciones =    "11111111111111111111"
    # alcance =        "11111111111111111111"
    # mecanismo =      "11111111111111111111"
    # estado_inicial = "10000000000000000000"
    # condiciones =    "11111111111111111111"
    # alcance =        "11011011011011011011"
    # mecanismo =      "10101010101010101010"

    gestor_sistema = Manager(estado_inicial)

    ### Ejemplo de solución mediante módulo de fuerza bruta ###
    analizador_fb = GeometricSIA(gestor_sistema)
    # analizador_fb = BruteForce(gestor_sistema)
    sia_uno = analizador_fb.aplicar_estrategia(
        condiciones,
        alcance,
        mecanismo,
    )
    print(sia_uno)
