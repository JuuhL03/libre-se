/**
 * Cria o formulário da pesquisa de avaliação do Libre-se (TCC, Grupo 6, Ciência da Computação - FIB)
 * e a planilha que recebe as respostas.
 *
 * Como usar: script.google.com → Novo projeto → apague o que estiver lá, cole este arquivo,
 * salve e clique em Executar (função criarFormulario). Na primeira vez o Google pede permissão
 * para o script criar formulários e planilhas na sua conta. Os links aparecem no
 * "Registro de execução" (e o formulário fica no seu Google Drive).
 */
const LINK_APP = "https://libre-se-web-production.up.railway.app/?pesquisa=1";
const CONTATO = "jullialopesgomes@gmail.com";

function criarFormulario() {
  const f = FormApp.create("Pesquisa Libre-se: teste do reconhecimento de Libras (TCC)");
  f.setDescription(
    "Pesquisa do Trabalho de Conclusão de Curso \"SignBridge / Libre-se\", do curso de Ciência da " +
    "Computação das Faculdades Integradas de Bauru (FIB), Grupo 6. Você vai testar um protótipo que " +
    "reconhece sinais de Libras pela câmera e responder algumas perguntas. Leva cerca de 10 a 15 minutos. " +
    "Pessoas surdas, intérpretes, estudantes de Libras e ouvintes podem participar.");
  f.setProgressBar(true);
  f.setCollectEmail(false);
  f.setAllowResponseEdits(false);
  f.setShowLinkToRespondAgain(false);
  f.setConfirmationMessage(
    "Obrigado por participar! Suas respostas ajudam a avaliar e melhorar o Libre-se.");

  // ------------------------------------------------------------------ 1. consentimento
  f.addSectionHeaderItem()
    .setTitle("Termo de consentimento")
    .setHelpText(
      "Objetivo: avaliar se o protótipo Libre-se reconhece sinais de Libras e se é fácil de usar.\n\n" +
      "O que você fará: abrir o app no navegador, permitir a câmera, fazer alguns sinais e responder " +
      "a este formulário.\n\n" +
      "Privacidade: a pesquisa é ANÔNIMA. Não pedimos nome, e-mail, foto nem vídeo. A imagem da câmera " +
      "não sai do seu aparelho: só os pontos das mãos (coordenadas numéricas) vão para o servidor, e os " +
      "sinais gravados no teste são apagados quando a sessão termina. As respostas serão usadas apenas " +
      "de forma agregada no TCC.\n\n" +
      "Riscos e benefícios: o risco é mínimo (cansaço leve ao repetir sinais). Não há pagamento. " +
      "A participação é voluntária e você pode desistir a qualquer momento, sem prejuízo, fechando a página.\n\n" +
      "Dúvidas: " + CONTATO);

  const consente = f.addMultipleChoiceItem()
    .setTitle("Você leu as informações acima e concorda em participar?")
    .setRequired(true);
  const idade = f.addMultipleChoiceItem()
    .setTitle("Você tem 18 anos ou mais?")
    .setRequired(true);

  // ------------------------------------------------------------------ 2. perfil
  const pPerfil = f.addPageBreakItem().setTitle("Sobre você");
  f.addMultipleChoiceItem()
    .setTitle("Você se identifica como:")
    .setChoiceValues(["Surdo(a)", "Pessoa com deficiência auditiva", "Ouvinte", "Prefiro não dizer"])
    .setRequired(true);
  f.addMultipleChoiceItem()
    .setTitle("Qual é a sua relação com a Libras?")
    .setChoiceValues([
      "Libras é a minha primeira língua",
      "Sou fluente em Libras",
      "Sou intérprete ou tradutor(a) de Libras",
      "Estou aprendendo Libras (nível básico ou intermediário)",
      "Não sei Libras"])
    .setRequired(true);
  f.addMultipleChoiceItem()
    .setTitle("Faixa etária")
    .setChoiceValues(["18 a 24 anos", "25 a 34 anos", "35 a 44 anos", "45 a 59 anos", "60 anos ou mais"])
    .setRequired(true);
  f.addMultipleChoiceItem()
    .setTitle("Você já usou algum aplicativo de tradução de Libras (por exemplo, Hand Talk ou VLibras)?")
    .setChoiceValues(["Sim", "Não"])
    .setRequired(true);

  // ------------------------------------------------------------------ 3. teste
  const pTeste = f.addPageBreakItem()
    .setTitle("Hora de testar")
    .setHelpText(
      "Abra o link abaixo em OUTRA ABA (no celular ou no computador) e permita a câmera:\n\n" +
      LINK_APP + "\n\n" +
      "O app mostra um roteiro com 3 tarefas:\n" +
      "1) ALFABETO: soletre L-I-B-R-A-S (e, se quiser, o seu primeiro nome).\n" +
      "2) CALIBRADO: o app pede oi, obrigado, café, pão e tchau, 3 vezes cada (é a calibração). " +
      "Não sabe algum sinal? Toque em \"Como se faz?\".\n" +
      "3) Depois da calibração, faça cada um dos 5 sinais 2 vezes e conte quantas vezes o app acertou.\n\n" +
      "Dica: luz de frente, a mão inteira aparecendo, faça o sinal e abaixe a mão. Quando terminar, " +
      "volte para esta aba e continue.");
  f.addMultipleChoiceItem()
    .setTitle("Em qual aparelho você fez o teste?")
    .setChoiceValues(["Celular Android", "iPhone", "Computador ou notebook", "Tablet"])
    .setRequired(true);
  f.addMultipleChoiceItem()
    .setTitle("Como estava a iluminação?")
    .setChoiceValues(["Luz natural (perto de janela)", "Luz de lâmpada", "Muito clara (sol ou luz forte)",
      "Pouca luz"])
    .setRequired(true);
  const abriu = f.addMultipleChoiceItem()
    .setTitle("Você conseguiu abrir o app e ver a sua mão na tela?")
    .setRequired(true);

  // ------------------------------------------------------------------ 4. tarefas
  const pTarefas = f.addPageBreakItem().setTitle("Resultado das tarefas");
  f.addScaleItem()
    .setTitle("Tarefa 1: quantas letras de L-I-B-R-A-S o app reconheceu certo?")
    .setBounds(0, 6)
    .setLabels("nenhuma", "todas as 6")
    .setRequired(true);
  f.addMultipleChoiceItem()
    .setTitle("Tarefa 1: você soletrou o seu nome?")
    .setChoiceValues(["Sim, saiu certo", "Sim, mas saiu com erros", "Não tentei"])
    .setRequired(true);
  f.addMultipleChoiceItem()
    .setTitle("Tarefa 2: você já sabia fazer os 5 sinais (oi, obrigado, café, pão, tchau)?")
    .setChoiceValues(["Sim, todos", "Alguns (vi os outros em \"Como se faz?\")", "Nenhum"])
    .setRequired(true);
  f.addScaleItem()
    .setTitle("Tarefa 2: gravar os sinais (calibração) foi...")
    .setBounds(1, 5)
    .setLabels("muito difícil", "muito fácil")
    .setRequired(true);
  f.addGridItem()
    .setTitle("Tarefa 3: em 2 tentativas, quantas vezes o app acertou cada sinal?")
    .setRows(["oi", "obrigado", "café", "pão", "tchau"])
    .setColumns(["0 de 2", "1 de 2", "2 de 2", "não fiz"])
    .setRequired(true);
  f.addParagraphTextItem()
    .setTitle("Quando o app errou, o que apareceu no lugar? (opcional)");
  f.addMultipleChoiceItem()
    .setTitle("Depois de abaixar a mão, a palavra apareceu...")
    .setChoiceValues(["Rápido (menos de 1 segundo)", "Em um tempo aceitável (1 a 3 segundos)",
      "Demorou (mais de 3 segundos)", "Não apareceu"])
    .setRequired(true);

  // ------------------------------------------------------------------ 5. SUS
  const pSus = f.addPageBreakItem()
    .setTitle("Facilidade de uso")
    .setHelpText("Marque de 1 (discordo totalmente) a 5 (concordo totalmente). " +
      "Questionário SUS (System Usability Scale).");
  [
    "Eu acho que gostaria de usar este sistema com frequência.",
    "Eu achei o sistema desnecessariamente complexo.",
    "Eu achei o sistema fácil de usar.",
    "Eu acho que precisaria da ajuda de uma pessoa com conhecimentos técnicos para usar o sistema.",
    "Eu achei que as várias funções do sistema estão bem integradas.",
    "Eu achei que o sistema apresenta muita inconsistência.",
    "Eu imagino que a maioria das pessoas aprenderia a usar este sistema rapidamente.",
    "Eu achei o sistema muito complicado de usar.",
    "Eu me senti confiante ao usar o sistema.",
    "Eu precisei aprender várias coisas novas antes de conseguir usar o sistema.",
  ].forEach((frase, i) => {
    f.addScaleItem()
      .setTitle(`${i + 1}. ${frase}`)
      .setBounds(1, 5)
      .setLabels("discordo totalmente", "concordo totalmente")
      .setRequired(true);
  });

  // ------------------------------------------------------------------ 6. opiniao
  const pOpiniao = f.addPageBreakItem().setTitle("Sua opinião");
  f.addScaleItem()
    .setTitle("Um app assim seria útil em situações do dia a dia (padaria, comércio, atendimento)?")
    .setBounds(1, 5)
    .setLabels("nada útil", "muito útil")
    .setRequired(true);
  f.addMultipleChoiceItem()
    .setTitle("Se você é surdo(a): usaria um app assim para conversar com quem não sabe Libras?")
    .setChoiceValues(["Sim", "Talvez", "Não", "Não se aplica (sou ouvinte)"])
    .setRequired(true);
  f.addCheckboxItem()
    .setTitle("O que mais atrapalhou? (marque quantas quiser)")
    .setChoiceValues(["O app reconheceu o sinal errado", "Demora para reconhecer", "A calibração foi cansativa",
      "Câmera ou iluminação", "A tela ou as instruções eram confusas", "Eu não sabia os sinais",
      "Nada atrapalhou"])
    .showOtherOption(true);
  f.addParagraphTextItem().setTitle("O que você mais gostou?");
  f.addParagraphTextItem().setTitle("O que falta ou o que você mudaria?");
  f.addParagraphTextItem().setTitle("Comentários livres (opcional)");

  // ------------------------------------------------------------------ 7. sem camera
  const pProblema = f.addPageBreakItem().setTitle("O app não abriu");
  f.addParagraphTextItem()
    .setTitle("O que aconteceu? (mensagem na tela, aparelho, navegador)")
    .setRequired(true);
  // setGoToPage na quebra de pagina vale para quem chega nela pela ordem normal, ou seja, quem
  // termina "Sua opinião": envia o formulário (só cai em "O app não abriu" quem respondeu Não)
  pProblema.setGoToPage(FormApp.PageNavigationType.SUBMIT);

  // ------------------------------------------------------------------ navegacao
  consente.setChoices([
    consente.createChoice("Sim, concordo em participar", FormApp.PageNavigationType.CONTINUE),
    consente.createChoice("Não", FormApp.PageNavigationType.SUBMIT)]);
  idade.setChoices([
    idade.createChoice("Sim", FormApp.PageNavigationType.CONTINUE),
    idade.createChoice("Não", FormApp.PageNavigationType.SUBMIT)]);
  abriu.setChoices([
    abriu.createChoice("Sim", pTarefas),
    abriu.createChoice("Sim, mas com dificuldade", pTarefas),
    abriu.createChoice("Não", pProblema)]);

  // ------------------------------------------------------------------ planilha das respostas
  const planilha = SpreadsheetApp.create("Respostas - Pesquisa Libre-se (TCC)");
  f.setDestination(FormApp.DestinationType.SPREADSHEET, planilha.getId());

  Logger.log("Formulário (editar): " + f.getEditUrl());
  Logger.log("Link para divulgar: " + f.getPublishedUrl());
  Logger.log("Planilha das respostas: " + planilha.getUrl());
}
