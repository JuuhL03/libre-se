// "Como se faz?": o video do sinal num dicionario de Libras (os mesmos links da versao web,
// libre-se/web/app.js, e do app de computador, calibracao.py link_dicionario).
// A calibracao grava como a PESSOA faz o sinal; quem ainda nao sabe aprende antes aqui.

import type { EstadoLibres } from './servidor/ClienteLibres';

const DICIONARIO_BUSCA = 'https://spreadthesign.com/pt.br/search/?q='; // Spread the Sign, Libras (pt.br)
const DICIONARIO_INES = 'https://dicionario.ines.gov.br/'; // Dicionario da Libras do INES
const DICIONARIO_ALFABETO = 'https://spreadthesign.com/pt.br/alphabet/';
const NUMEROS_EXTENSO = [
  'zero', 'um', 'dois', 'três', 'quatro', 'cinco', 'seis', 'sete', 'oito', 'nove', 'dez',
];

/** "BOM_DIA" -> "bom dia"; "2" -> "dois" (o dicionario procura numeros por extenso). */
export function termoDicionario(p: string): string {
  const t = String(p).replace(/_/g, ' ').trim().replace(/\s+/g, ' ').toLowerCase();
  return /^\d+$/.test(t) && +t < NUMEROS_EXTENSO.length ? NUMEROS_EXTENSO[+t] : t;
}

export const linkDicionario = (p: string) =>
  DICIONARIO_BUSCA + encodeURIComponent(termoDicionario(p));

/**
 * O link do "Como se faz?" para o momento: no alfabeto, o alfabeto manual; calibrando,
 * a palavra da vez; fora disso, o ultimo sinal (reconhecido ou o mais parecido).
 */
export function linkComoSeFaz(estado: EstadoLibres | null): string {
  if (!estado) return DICIONARIO_INES;
  if (estado.modo === 'ALFABETO') return DICIONARIO_ALFABETO;
  if (estado.cal?.calibrando && estado.cal.palavra) return linkDicionario(estado.cal.palavra);
  const ultimo = estado.ultimo?.ok || estado.ultimo?.melhor;
  return ultimo ? linkDicionario(ultimo) : DICIONARIO_INES;
}
