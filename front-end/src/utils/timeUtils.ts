export function convertToKSTFromUTC(utcString: string): string {
  const isoString = utcString.replace(' ', 'T') + 'Z';
  const date = new Date(isoString);

  const kstTime = new Date(date.getTime() + 9 * 60 * 60 * 1000);

  const yyyy = kstTime.getUTCFullYear();
  const mm = String(kstTime.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(kstTime.getUTCDate()).padStart(2, '0');
  const hh = String(kstTime.getUTCHours()).padStart(2, '0');
  const mi = String(kstTime.getUTCMinutes()).padStart(2, '0');
  const ss = String(kstTime.getUTCSeconds()).padStart(2, '0');
  const ms = String(kstTime.getUTCMilliseconds()).padStart(3, '0');

  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}.${ms}`;
}

export function convertToKSTFromTimestamp(timestamp: number): string {
  const date = new Date(timestamp + 9 * 60 * 60 * 1000); // UTC → KST

  const yyyy = date.getUTCFullYear();
  const mm = String(date.getUTCMonth() + 1).padStart(2, '0');
  const dd = String(date.getUTCDate()).padStart(2, '0');
  const hh = String(date.getUTCHours()).padStart(2, '0');
  const mi = String(date.getUTCMinutes()).padStart(2, '0');
  const ss = String(date.getUTCSeconds()).padStart(2, '0');
  const ms = String(date.getUTCMilliseconds()).padStart(3, '0');

  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}.${ms}`;
}
