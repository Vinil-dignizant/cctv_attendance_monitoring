# face_recognition.spec
block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=['D:\\vinil\\face_recognition3'],
    binaries=[],
    datas=[
        ('config', 'config'),
        ('snapshots', 'snapshots'),
        ('assets', 'assets')
    ],
    hiddenimports=[
        'face_detection.scrfd',
        'face_recognition.arcface'
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher
)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='FaceRecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    icon='assets\\icon.ico'
)