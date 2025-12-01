# ANÁLISE DE SUCESSO DE JOGOS STEAM: INDIES vs AAA

**Data da análise:** 01/12/2025 20:28
**Total de jogos analisados:** 20000
**Período:** 2015.0 - 2025.0

##RESUMO EXECUTIVO

- **Jogos Indies analisados:** 18659 (93.3%)
- **Taxa de sucesso Indies:** 52.04%
- **Taxa de sucesso AAA:** 21.63%
- **Indies têm +30.41% mais chance de sucesso**

##RESULTADOS DOS MODELOS

| Modelo | Acurácia | Precisão | Recall | F1-Score | CV F1-Score |
|--------|----------|----------|--------|----------|-------------|
| Random Forest | 0.6767 | 0.6758 | 0.6790 | 0.6774 | 0.6846 (±0.0036) |
| Naive Bayes | 0.5840 | 0.5534 | 0.8710 | 0.6768 | 0.6731 (±0.0065) |

**Melhor modelo:** Random Forest
**F1-Score:** 0.6774

##CONCLUSÕES PRINCIPAIS

### Fatores Mais Importantes para o Sucesso:

1. **achievements** (importância: 0.2240)
1. **price** (importância: 0.1558)
1. **cat_steam_achievements** (importância: 0.1217)
1. **num_supported_languages** (importância: 0.1115)
1. **cat_full_controller_support** (importância: 0.0516)

### Recomendações para Desenvolvedores:

1. **Preço adequado é crucial** - Indies devem manter preços entre $10-30
2. **Multiplataforma aumenta chances** - Lançar para múltiplas plataformas
3. **Conquistas importam** - Implemente um sistema de conquistas
4. **Identidade clara** - Ser identificado como Indie pode ser vantajoso
5. **Evitar jogos gratuitos** - Jogos pagos têm maior taxa de sucesso

### Limitações do Estudo:

1. **Dados históricos** - O mercado de jogos evolui rapidamente
2. **Definição de sucesso** - Baseada em critérios simplificados
3. **Features disponíveis** - Limitado a dados públicos do Steam
4. **Viés de sobrevivência** - Apenas jogos que foram lançados
5. **Desbalanceamento** - Muitos mais Indies que AAA

## VISUALIZAÇÕES

![Análise Completa](figures/analise_completa.png)

---
*Relatório gerado automaticamente pelo sistema de análise de dados*