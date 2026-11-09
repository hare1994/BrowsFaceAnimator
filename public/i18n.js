// English is the key; Hebrew translations below. t() returns the text for the current language.
const HE = {
  'Brow Mapping Studio': 'סטודיו למיפוי גבות',
  'Front-facing photo in. Brows removed, mapped and redrawn from her own facial measurements.': 'מעלים תמונה מלפנים: הגבות נמחקות, ממופות ומצוירות מחדש לפי מידות הפנים שלה.',
  'Choose a photo': 'בחירת תמונה',
  'or drop it here': 'או לגרור אותה לכאן',
  'Head only, facing the camera': 'רק הראש, מול המצלמה',
  'Even light, hair off the forehead': 'תאורה אחידה, שיער מחוץ למצח',
  'Your photo never leaves this device': 'התמונה לא יוצאת מהמכשיר',
  'Replay': 'הפעלה מחדש',
  'Hold: original': 'לחיצה: מקור',
  'Shape': 'צורה',
  'Thickness': 'עובי',
  'Arch height': 'גובה קשת',
  'Arch position': 'מיקום קשת',
  'Gap': 'רווח',
  'Gap between brows': 'רווח בין הגבות',
  'Tail': 'זנב',
  'Tail length': 'אורך זנב',
  'Colour & style': 'צבע וסגנון',
  'Colour': 'צבע',
  'Style': 'סגנון',
  'Natural (photo-real)': 'טבעי (ריאליסטי)',
  'Hair strokes': 'שערות מצוירות',
  'Hair strokes + powder': 'שערות + פודרה',
  'Powder': 'פודרה',
  'Measurements': 'מידות',
  'Her measurements': 'המידות שלה',
  'mm values assume an average iris width of 11.7 mm.': 'המידות במ"מ מניחות רוחב קשתית ממוצע של 11.7 מ"מ.',
  'Save image': 'שמירת תמונה',
  'Save animation': 'שמירת סרטון',
  'Saving…': 'שומרת…',
  'New photo': 'תמונה חדשה',
  // measurements
  'Brow start from centre': 'תחילת הגבה מהמרכז',
  'Arch from centre': 'הקשת מהמרכז',
  'Tail from centre': 'הזנב מהמרכז',
  'Brow length': 'אורך הגבה',
  'Thickness at start': 'עובי בתחילת הגבה',
  'Arch lift': 'הרמת הקשת',
  'Gap above pupil': 'מרחק מעל האישון',
  'Face symmetry': 'סימטריית הפנים',
  // animation captions
  'Centre line': 'קו אמצע',
  'Start: inner nose line, straight up': 'התחלה: קו פנימי של האף, ישר למעלה',
  'Brow height': 'גובה הגבה',
  'Arch: above the outer edge of the iris': 'קשת: מעל הקצה החיצוני של הקשתית',
  'Tail: nostril past the outer eye corner': 'זנב: מהנחיר, אחרי זווית העין',
  'Mapping points': 'נקודות מיפוי',
  'Connecting the shape': 'מחברת את הצורה',
  'Filling in': 'ממלאת',
  'Analysing the face': 'מנתחת את הפנים',
  'Erasing the brows': 'מוחקת את הגבות',
  // working animation
  'Done': 'מוכן',
  'Getting ready': 'מתכוננת',
  'Warming up the face model': 'מכינה את מודל הפנים',
  'Loading the skin AI': 'טוענת את ה-AI של העור',
  'Finding her face': 'מאתרת את הפנים',
  'Locating 478 face points': 'מאתרת 478 נקודות בפנים',
  'Mapping brow hairs': 'ממפה את שערות הגבה',
  'Tracing every hair': 'עוקבת אחרי כל שערה',
  'Protecting lids and eyeshadow': 'שומרת על העפעף והצללית',
  'Downloading the skin AI': 'מורידה את ה-AI של העור',
  'One-time download (~28 MB)': 'הורדה חד-פעמית (כ-28MB)',
  'Saved on this device for next time': 'נשמר במכשיר לפעם הבאה',
  'Erasing brow 1 of 2': 'מוחקת גבה 1 מתוך 2',
  'Erasing brow 2 of 2': 'מוחקת גבה 2 מתוך 2',
  'Rebuilding the skin underneath': 'בונה מחדש את העור',
  'Matching skin tone': 'מתאימה את גוון העור',
  'Recreating pores and texture': 'משחזרת נקבוביות ומרקם',
  'Blending with the forehead': 'ממזגת עם המצח',
  'Keeping the eyelid untouched': 'משאירה את העפעף כמו שהוא',
  'Softening the edges': 'מרככת את הקצוות',
  'Polishing': 'מלטשת',
  'Checking for any leftover hairs': 'בודקת שלא נשארו שערות',
  // messages
  'No face found. Try a well-lit, front-facing photo.': 'לא נמצאו פנים. כדאי לנסות תמונה מוארת, מלפנים.',
  'Something went wrong: ': 'משהו השתבש: ',
  'Video recording is not supported in this browser.': 'הדפדפן הזה לא תומך בהקלטת סרטון.',
  'AI erase failed, so a basic erase was used. Reason: ': 'מחיקת ה-AI נכשלה, ולכן נעשתה מחיקה בסיסית. הסיבה: ',
  '. Open this page with ?diag=1 at the end of the address for a full check.': '. לבדיקה מלאה אפשר לפתוח את העמוד עם ‎?diag=1 בסוף הכתובת.',
};

let lang = 'en';
try { const s = localStorage.getItem('brow-lang'); if (s === 'he' || s === 'en') lang = s; else if ((navigator.language || '').startsWith('he')) lang = 'he'; } catch {}

export const getLang = () => lang;
export const t = (s) => (lang === 'he' && HE[s]) || s;

const listeners = [];
export const onLangChange = (fn) => listeners.push(fn);

export function applyLang() {
  document.documentElement.lang = lang;
  document.documentElement.dir = lang === 'he' ? 'rtl' : 'ltr';
  document.title = t('Brow Mapping Studio');
  for (const el of document.querySelectorAll('[data-i18n]')) el.textContent = t(el.dataset.i18n);
  const btn = document.getElementById('lang');
  if (btn) btn.textContent = lang === 'he' ? 'English' : 'עברית';
  for (const fn of listeners) fn(lang);
}
export function setLang(l) {
  lang = l;
  try { localStorage.setItem('brow-lang', l); } catch {}
  applyLang();
}
