# Gabarito válido apenas para PARACETAMOL-Bula-Profissional.pdf (fabricante
# Geolab). Outra bula — mesmo de paracetamol, de outro fabricante — pode ter
# seções em páginas diferentes; crie um novo arquivo neste diretório para
# cada documento avaliado, escrito manualmente a partir do PDF real (gerar
# o gabarito com uma LLM introduziria viés circular na avaliação).
from app.evaluation.caso_de_teste import CasoDeTeste

DATASET = [
    CasoDeTeste(
        pergunta="Para que serve este medicamento?",
        resposta_esperada=(
            "Redução de febre e alívio temporário de dores leves a moderadas, "
            "em adultos e em crianças."
        ),
        pagina_esperada="2",  # Seção 1. INDICAÇÕES
    ),
    CasoDeTeste(
        pergunta="Quais são as contraindicações deste medicamento?",
        resposta_esperada=(
            "Não deve ser administrado a pacientes com hipersensibilidade ao "
            "paracetamol ou a qualquer outro componente da fórmula."
        ),
        pagina_esperada="4",  # Seção 4. CONTRAINDICAÇÕES
    ),
    CasoDeTeste(
        pergunta="Como devo armazenar este medicamento?",
        resposta_esperada=(
            "Manter em temperatura ambiente (15°C a 30°C), protegido da luz e da umidade."
        ),
        pagina_esperada="6",  # Seção 7. CUIDADOS DE ARMAZENAMENTO — o falso negativo conhecido!
    ),
    CasoDeTeste(
        pergunta="Como devo administrar a dose em gotas para uma criança?",
        resposta_esperada=(
            "1 gota por kg de peso, até o máximo de 35 gotas, para crianças abaixo de "
            "12 anos. Para crianças abaixo de 11kg ou 2 anos, é necessário consultar um médico."
        ),
        pagina_esperada="6",  # Seção 8. POSOLOGIA E MODO DE USAR
    ),
    CasoDeTeste(
        pergunta="Quais são as reações adversas deste medicamento?",
        resposta_esperada=(
            "Reações muito raras: urticária, coceira, vermelhidão no corpo, reações "
            "alérgicas e aumento das transaminases."
        ),
        pagina_esperada="7",  # Seção 9. REAÇÕES ADVERSAS
    ),
    CasoDeTeste(
        pergunta="Este medicamento trata infecções bacterianas?",
        resposta_esperada=(
            "A bula não aborda esse assunto — a resposta correta é dizer que essa "
            "informação não foi encontrada, nunca inventar uma resposta."
        ),
        pagina_esperada=None,  # pergunta fora do escopo da bula, de propósito
    ),
]
