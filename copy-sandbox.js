const fs = require('fs');
const path = require('path');

// Ensure dist directory exists
const distDir = path.join(__dirname, 'dist');
if (!fs.existsSync(distDir)) {
  fs.mkdirSync(distDir, { recursive: true });
}

// Copy document sandbox code
const source = path.join(__dirname, 'document-sandbox', 'code.js');
const dest = path.join(distDir, 'code.js');

try {
  fs.copyFileSync(source, dest);
  console.log('✓ Document sandbox copied to dist/code.js');
} catch (error) {
  console.error('Error copying document sandbox:', error);
  process.exit(1);
}
