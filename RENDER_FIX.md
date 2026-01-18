# Fix for Render Deployment - Python Version Issue

## Problem
Render is using Python 3.13.4 by default, but pandas 2.1.4 doesn't support Python 3.13 yet.

## Solution
I've created `runtime.txt` file that specifies Python 3.12.0.

## Steps to Fix

### Option 1: Use runtime.txt (Recommended)

The `runtime.txt` file has been created with:
```
python-3.12.0
```

Render will automatically detect and use this file.

### Option 2: Set in Render Dashboard

1. Go to your Render service dashboard
2. Click on "Environment" tab
3. Add environment variable:
   - Key: `PYTHON_VERSION`
   - Value: `3.12.0`
4. Save and redeploy

### Option 3: Update render.yaml

If you're using `render.yaml`, I've updated it to include:
```yaml
runtime: python-3.12.0
```

## After Fixing

1. Commit the `runtime.txt` file:
   ```bash
   cd backend
   git add runtime.txt render.yaml
   git commit -m "Specify Python 3.12 for compatibility"
   git push
   ```

2. Render will automatically redeploy with Python 3.12

3. The build should now succeed!

## Verify

After deployment, check the build logs:
- Should see: "Installing Python version 3.12.0..."
- Should NOT see: "Installing Python version 3.13.4..."

## Alternative: Update to Newer pandas

If you want to use Python 3.13, update pandas:

```txt
pandas>=2.2.0
```

But Python 3.12 is more stable and recommended for now.
