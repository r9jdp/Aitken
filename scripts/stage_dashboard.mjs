// Stage only the compiled public output for the static host. Never copy model inputs.
import { cpSync, existsSync, mkdirSync, readdirSync, lstatSync, realpathSync } from 'node:fs';
import { resolve, dirname, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const source = resolve(root, 'dashboard/dist/client');
const target = resolve(root, 'dist');
if (!existsSync(resolve(source, 'index.html'))) throw new Error('Build the dashboard first: npm --prefix dashboard run build');
if (existsSync(target) && (lstatSync(target).isSymbolicLink() || !realpathSync(target).startsWith(realpathSync(root)))) throw new Error('Unsafe staging directory');
mkdirSync(target, { recursive: true });
for (const item of readdirSync(source)) {
  if (item === '.vite' || item === '.openai' || item === 'vinext-client-entry-manifest.json') continue;
  cpSync(resolve(source, item), resolve(target, item), { recursive: true });
}
console.log(`Staged public dashboard in ${relative(root, target)}. No raw observations or checkpoints included.`);
