# Acceso a la VM H100 vía túnel Cloudflare — Guía de configuración

Guía para que el equipo (Aaron, Isaac) acceda por **SSH** a la VM H100 del sponsor
(`gjcamacho-gpuh1`) para correr training/scripts. El acceso es por túnel Cloudflare
porque la VM es Windows sin cuenta Azure para nosotros y el NSG no expone el puerto SSH.

> **URL del túnel ACTUAL**: se comparte por el canal privado del equipo (NO se versiona
> en el repo). Formato: `ejemplo-palabras-aleatorias.trycloudflare.com`.
> ⚠️ La URL **cambia cada vez que la VM se reinicia**. Pedir la URL vigente a quien tenga
> acceso a la VM (ver sección 5: "Cuando cambia la URL").
>
> 🔒 **Seguridad**: este documento es público (se versiona). NO pegar aquí la URL real,
> ni IPs, ni el puerto SSH concreto. Esos datos van solo en el canal privado del equipo.

---

## 0. Cómo funciona (resumen)

```
Tu PC ──ssh──► cloudflared access (tu PC) ──► Cloudflare ──► cloudflared (VM) ──► sshd:<PUERTO-SSH> (VM)
```

- La VM corre un `cloudflared` que sale a Cloudflare y expone su SSH (puerto <PUERTO-SSH>).
- En **tu** PC corres otro `cloudflared` que mapea ese túnel a tu `127.0.0.1:<PUERTO-SSH>`.
- Luego haces `ssh ... User1@127.0.0.1 -p <PUERTO-SSH>` y entras a la VM.
- Autenticación por **llave SSH** (cada quien la suya), no por contraseña.

> ⚠️ **Usa `127.0.0.1`, NUNCA `localhost`.** En Windows el cliente OpenSSH resuelve
> `localhost` a `::1` (IPv6), pero `cloudflared access tcp` solo escucha en `127.0.0.1`
> (IPv4): con `localhost` el SSH falla con `banner exchange: Connection refused` aunque
> la VM esté sana. Con `127.0.0.1` explícito conecta. (Ver Troubleshooting.)

---

## 0.1 Caso del sponsor (acceso remoto vía túnel con su llave)

El sponsor es **dueño y administrador** de la VM (`gjcamacho-gpuh1`), así que tiene una
ventaja sobre el resto del equipo: para entrar por SSH desde fuera **no depende de nadie**
—él mismo autoriza su llave en la VM. El flujo completo, de principio a fin:

1. **En su PC** (el que usará para conectarse remoto): instalar `cloudflared` (sección 1.1)
   y generar su par de llaves SSH (sección 1.2), p. ej. `-C "sponsor-h100"`.
2. **Autorizar su propia llave pública en la VM**: como es admin, lo hace él directamente
   en la VM (por RDP/consola) — es la sección 4, pero aplicada a su propia `.pub`. No
   necesita enviársela a nadie.
3. **Conectarse**: levantar el túnel local + `ssh` exactamente como el resto del equipo
   (sección 2), usando la URL vigente del túnel (sección 5) y `User1@127.0.0.1`.

> 💡 El sponsor podría entrar por **RDP/consola local** sin túnel (es su máquina). El túnel
> SSH solo le hace falta si quiere acceso **remoto por línea de comandos** desde otro PC,
> igual que el equipo. Esta guía cubre ese caso (SSH remoto).

> 🔑 Si en lugar de **conectarse él** quiere **dar acceso a un cuarto colaborador**, no
> hace falta tocar nada nuevo: ese colaborador sigue las secciones 1–2 y el sponsor (o
> Arthur) agrega su `.pub` con la sección 4.

---

## 1. Requisitos (una sola vez)

### 1.1 Instalar cloudflared en tu PC

**Windows (PowerShell):**
```powershell
$dir = "$env:USERPROFILE\.cloudflared"
New-Item -ItemType Directory -Force -Path $dir | Out-Null
Invoke-WebRequest -Uri "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe" -OutFile "$dir\cloudflared.exe" -UseBasicParsing
& "$dir\cloudflared.exe" --version
```

**macOS / Linux:**
```bash
# macOS
brew install cloudflared
# Linux (debian/ubuntu)
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -o ~/.local/bin/cloudflared && chmod +x ~/.local/bin/cloudflared
cloudflared --version
```

### 1.2 Generar tu llave SSH

```bash
ssh-keygen -t ed25519 -f "$HOME/.ssh/agrosat_h100" -C "TU-NOMBRE-h100"
```
- Usa tu nombre en `-C` (ej. `aaron-h100`, `isaac-h100`).
- Passphrase: opcional (tu decides).

### 1.3 Pasar tu llave PÚBLICA al admin de la VM

Manda **solo** el contenido de tu llave **pública** (la `.pub`) a quien administra la VM
(Arthur). **Nunca** mandes la privada.

```bash
cat "$HOME/.ssh/agrosat_h100.pub"
# Copia esa linea: empieza con "ssh-ed25519 AAAA... TU-NOMBRE-h100"
```

El admin la agrega al `administrators_authorized_keys` de la VM (ver sección 4).

---

## 2. Conectarse (cada vez)

Una vez tu llave pública está autorizada en la VM:

**Terminal 1 — abre el túnel local (déjala abierta):**
```bash
cloudflared access tcp --hostname <URL-VIGENTE>.trycloudflare.com --url localhost:<PUERTO-SSH>
```
(En Windows: `& "$env:USERPROFILE\.cloudflared\cloudflared.exe" access tcp --hostname <URL>.trycloudflare.com --url localhost:<PUERTO-SSH>`)

**Terminal 2 — conéctate por SSH (usa `127.0.0.1`, no `localhost`):**
```bash
ssh -p <PUERTO-SSH> -i ~/.ssh/agrosat_h100 -o IdentitiesOnly=yes User1@127.0.0.1
```

Si entra, verás el prompt de la VM (`gjcamacho-gpuh1`).

### Trabajar con el entorno de la VM

```powershell
# El entorno Python del proyecto se usa via micromamba:
F:\tools\micromamba.exe run -n agrosat python --version
# Repo:
cd F:\projects\agrosat-copilot
# Verificar GPU:
nvidia-smi
F:\tools\micromamba.exe run -n agrosat python -c "import torch; print(torch.cuda.is_available())"
```

---

## 3. Notas importantes

- **Todos entran como `User1` (admin).** No hay aislamiento entre usuarios: los tres
  comparten el mismo usuario y ven los mismos archivos. Coordinar para no pisarse
  (ej. cada quien su carpeta de trabajo, su run de MLflow).
- **No apagar la VM** (el sponsor la paga 24/7). No correr `Restart-Computer` sin avisar
  al equipo (cambia la URL del túnel para todos).
- **Disco**: trabajar en **F:** (3.8 TB). C: tiene ~11 GB, no usar.
- **GPU**: H100 NVL 96 GB, driver 596.36, CUDA 13.2. Una sola GPU -> coordinar para no
  lanzar dos trainings pesados a la vez (revisar `nvidia-smi` antes).

---

## 4. (Solo admin) Autorizar la llave de un compañero

En la VM, PowerShell **Administrador**:
```powershell
$pub = 'ssh-ed25519 AAAA...PEGAR-LA-PUBLICA... nombre-h100'
$adminKeys = "$env:ProgramData\ssh\administrators_authorized_keys"
Add-Content -Path $adminKeys -Value $pub
# Reaplicar permisos (obligatorio, si no sshd ignora el archivo):
icacls $adminKeys /inheritance:r /grant "Administrators:F" /grant "SYSTEM:F"
Get-Content $adminKeys   # verificar
```

Para **revocar** a alguien: editar ese archivo y borrar su línea.

---

## 5. Cuando cambia la URL (tras reinicio de la VM)

La URL `*.trycloudflare.com` es efímera. Tras un reinicio, leerla en la VM:
```powershell
Select-String -Path C:\cloudflared\tunnel.log -Pattern "trycloudflare.com" | Select-Object -Last 1
```
Compartir la nueva URL al equipo **por el canal privado** (NO en este documento).

El túnel se levanta solo al arrancar la VM (Tarea Programada `AgroSatTunnel`), no hay que
hacer nada manual en la VM salvo leer la URL.

### Solución definitiva (pendiente, opcional)
Para una **URL fija** que no cambie nunca: registrar un dominio (~$10/año) en Cloudflare
y usar un *named tunnel* con Public Hostname SSH. Elimina el paso de re-compartir URL.

---

## 6. Troubleshooting

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| `Connection reset by peer` | URL del túnel vieja (la VM reinició) | Pedir URL nueva (sección 5) |
| `banner exchange: Connection refused` | Conectaste a `localhost` (IPv6 `::1`) en vez de `127.0.0.1` | Usar `User1@127.0.0.1` (ver §0) |
| `Permission denied (publickey)` | Tu llave no está autorizada | Reenviar tu `.pub` al admin (sponsor: autorízala tú, §4) |
| `cloudflared: command not found` | No instalado | Sección 1.1 |
| SSH cuelga sin conectar | Túnel local (Terminal 1) no está abierto | Abrir el `cloudflared access tcp` primero |
| Warning "post-quantum key exchange" | Cosmético (versión sshd) | Ignorar |
