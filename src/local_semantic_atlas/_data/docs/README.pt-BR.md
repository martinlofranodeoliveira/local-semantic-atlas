# Local Semantic Atlas

Busca neural multilíngue local com revisão fixa do modelo, entrada limitada e proveniência verificável. O encoder E5 aprovado produz os embeddings reais; o SDK calcula a média mascarada, normaliza os vetores e classifica por similaridade de cosseno em float64. O projeto não gera respostas. Escores são similaridades relativas, não probabilidades nem garantias de relevância.

[English](../README.md)

A qualificação observada usa Python 3.13.18, NumPy 2.5.4, ONNX Runtime 1.31.1 e tokenizers 0.23.3 em CPU Linux x86_64. A variante upstream é qint8 `avx512_vnni`. Outras CPUs, plataformas e versões não foram qualificadas; o nome da variante não garante portabilidade.

## Preparação explícita

Use o fluxo Docker em duas etapas abaixo ou instale o wheel após restaurar o lock exato do runtime. O comando instalado `local-semantic-atlas` usa o mesmo SDK limitado.

```sh
local-semantic-atlas prepare-model --destination ./approved-model
```

Somente esse comando usa rede. Ele busca seis URLs HTTPS fixas da revisão `614241f622f53c4eeff9890bdc4f31cfecc418b3` de `intfloat/multilingual-e5-small`, verifica o tamanho e SHA-256 de cada arquivo e mantém o README e as configurações upstream junto dos dois artefatos usados pelo runtime. Não aceita URL, revisão, provedor nem autenticação alternativos. Redirecionamentos HTTPS da CDN são permitidos; downgrade de TLS é rejeitado e proxies de ambiente não são usados. Cada arquivo tem prazo verificado de 300 segundos, acrescido de no máximo um intervalo bloqueante de leitura de 20 segundos.

O destino precisa ser novo e seu diretório pai deve ser controlado pelo operador. A criação exclusiva usa modo 0700. Durante a transferência, o diretório é staging privado. Após verificar, fechar e executar fsync em todos os arquivos, o programa publica `READY.json` com link atômico sem substituição. O manifesto lista os seis hashes e tamanhos; indica conclusão, não assinatura. Falhas retornam código diferente de zero e deixam o staging para inspeção ou remoção manual. Não há retomada, sobrescrita de diretório existente ou remoção de dados do usuário. Se o fsync final do diretório falhar, o marcador completo pode estar presente apesar da saída diferente de zero; registre conjuntamente o status do comando e o manifesto verificado.

Os arquivos somam cerca de 136 MB. Pesos e wheels são obtidos separadamente sob suas licenças upstream e não são o código-fonte deste projeto. O downloader tem testes com respostas controladas e uma nova transferência real observada: os seis artefatos passaram nos limites de bytes e hashes. Os snapshots recém-preparados alimentam os gates offline do pacote/runtime.

## Busca com arquivos literais

```sh
local-semantic-atlas search \
  --documents examples/documents.json \
  --query-file examples/query.pt.txt \
  --model approved-model/model_qint8_avx512_vnni.onnx \
  --tokenizer approved-model/tokenizer.json
```

O JSON de documentos aceita exatamente `schema_version: 1` e `documents`, uma lista de objetos com apenas strings `id` e `text`. Chaves duplicadas, números decimais, campos extras e mais de 256 registros são rejeitados. IDs são únicos, ASCII, com até 64 bytes e padrão `[A-Za-z0-9][A-Za-z0-9_.-]{0,63}`. Os textos são literais: sem trim, normalização Unicode, divisão em trechos ou truncamento. Uma LF final no arquivo de consulta integra a consulta. Arquivos precisam ser regulares; symlink final, FIFO e dispositivo são rejeitados. A leitura captura bytes próprios limitados e verifica mudanças observadas de identidade, tamanho e metadados; não é sandbox de filesystem hostil nem bloqueio persistente do caminho.

A busca não baixa modelos. A CLI verifica tamanhos durante a leitura; o SDK verifica ambos os hashes antes de construir tokenizador ou sessão nativa. Cada execução reconstrói o índice em memória e codifica a consulta. Não salva vetores nem mantém cache persistente. O JSON de stdout contém IDs classificados, escores, contagens e proveniência do modelo/corpus. Omite os textos e caminhos. IDs e digest do corpus ainda podem identificar um conjunto de dados: trate o relatório de acordo com esse contexto.

```sh
local-semantic-atlas search \
  --documents examples/documents.json --query-file examples/query.pt.txt \
  --model approved-model/model_qint8_avx512_vnni.onnx \
  --tokenizer approved-model/tokenizer.json \
  --format html > report.pt.html
```

O HTML usa PT-BR por padrão; `--locale en` seleciona inglês. É um documento estático imprimível, com tabela semântica, estado vazio/erro e lista de proveniência. Não carrega scripts, imagens, fontes ou recursos externos. O esquema fechado do SDK valida os dados antes do escape HTML. Não foram observados layout em navegador, tecnologia assistiva ou conformidade de acessibilidade; as verificações são de fonte e saída nativa.

## Limites e saídas

| Fronteira | Máximo |
| --- | --- |
| Documentos / hits | 256 / `min(corpus_count, top_k)` |
| Cada consulta ou documento | 8.192 bytes UTF-8 estritos; não vazio, sem NUL/surrogates |
| Soma dos textos dos documentos | 1.048.576 bytes UTF-8 |
| Arquivo JSON de documentos | 2.097.152 bytes |
| Passagem/consulta com prefixo e tokens especiais | 256 tokens; excesso rejeitado antes da inferência |
| Orçamento de tokens do índice | 65.536 tokens |
| Relatório JSON / HTML | 262.144 bytes UTF-8 cada, incluindo LF final |

Padrões: `--top-k 5` (1–256), `--max-documents 256` (0–256) e `--max-tokens 65536` (0–65.536). O orçamento limita separadamente cada operação de indexação/busca; não é uma soma compartilhada entre ambas. Mesmo com corpus vazio, uma consulta admitida passa pelo modelo antes de `COMPLETE_EMPTY`. Esgotamento não publica índice ou hits parciais.

| Código de saída | Significado |
| --- | --- |
| 0 | `COMPLETE` ou `COMPLETE_EMPTY`; ajuda/preparação bem-sucedidas também usam 0 |
| 2 | Entrada, artefato ou uso do comando inválido |
| 3 | Operação incompleta ou falha de entrada/saída/preparação |
| 4 | Resultado numericamente indeterminado |

Erros retornam códigos estáveis e JSON fechado, sem texto da exceção ou traceback. O status original da operação determina a saída; serializar um erro com sucesso não torna a operação bem-sucedida. A serialização completa ocorre antes da escrita. Falha de stdout retorna 3 e tenta emitir diagnóstico. O sistema pode ter aceitado um prefixo antes de falhar: descarte stdout se a saída não for 0. O redirecionamento do shell pode criar/truncar um destino mesmo quando o comando falha.

Os eventos de telemetria do ORT são desabilitados antes da criação da sessão. A biblioteca nativa ainda pode emitir aviso sobre persistência do identificador em filesystem somente leitura durante o import; não se afirma ausência total de comportamento de telemetria no import. As verificações reais usam rede desabilitada, configurações de thread do modelo e limites explícitos de recursos. Não se afirma igualdade bit a bit entre hardwares.

## Evidência

Os 40 checks incluem os 15 checks atuais do core, os 8 originais da fronteira de arquivos e 17 adicionais. A CLI foi executada com o modelo aprovado real em JSON e HTML nos dois idiomas, incluindo corpus vazio, orçamento insuficiente, UTF-8 inválido e inferência com stdout/stderr em falha. A integração de fonte antecedeu os gates separados do pacote instalado. Não é benchmark de qualidade. Evidência local e CI hospedada na fonte exata são distintas; revisão independente e CI hospedada não são inferidas dos checks locais.


## Fluxo do pacote

Execute `sh scripts/ci.sh`. O `Dockerfile.prepare` usa rede explicitamente para obter os 24 wheels exatos e baixar os seis artefatos fixos. O `Dockerfile` separado usa `docker build --network none` sobre a imagem local concluída, constrói sdist/wheel reais, verifica o pacote e reconstrói um wheel a partir do sdist. O runtime instala apenas os 20 itens do lock de runtime e esse wheel. Não contém checkout, fonte da preparação, compilador, frontend ou backend de build. O contêiner final sem binds executa os checks, o consumidor numérico escalar independente e a CLI instalada sob os limites do script. A imagem preparada precisa existir antes do build offline; não se presume cache entre jobs.

Instalação manual: `python -m pip install --require-hashes -r requirements-runtime.lock`, depois `python -m pip install --no-deps dist/local_semantic_atlas-0.1.0-py3-none-any.whl`. Para instalação offline acrescente `--no-index --find-links=...` apontando ao wheelhouse preparado explicitamente. Ferramentas de build usam `requirements-dev.lock`. Hashes binários cobrem apenas o tuple Linux x86_64/Python3.13 qualificado.

Consulte [segurança](../SECURITY.md), [contribuição](../CONTRIBUTING.md) e [proveniência](../NOTICE.md). Código original MIT; termos upstream separados. Publicação de [LOFRANO TECNOLOGIA](https://lofranotecnologia.com.br).


### Exemplo literal observado

O arquivo `examples/query.pt.txt` inclui LF final, preservada na consulta. No host qualificado, a CLI instalada retornou `COMPLETE`, 99 tokens de documentos e 20 da consulta, ordenando `stock` (0.8803328171865816), `maintenance` (0.7418338083175413) e `backup` (0.7131352326394726). O [JSON observado completo](observed-demo.json) registra os hashes. São exemplos sintéticos originais de manutenção/estoque/backup, não benchmark de qualidade. Remover a LF altera a consulta literal e pode mudar embeddings e escores; não se promete estabilidade bit a bit entre hardwares.
