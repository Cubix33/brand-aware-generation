import addOnSandboxSdk from "add-on-sdk-document-sandbox";

console.log("Document sandbox loading...");

addOnSandboxSdk.ready.then(async () => {
    console.log("✓ Document sandbox SDK is ready");

    // Listen for assets from the UI
    addOnSandboxSdk.instance.on("addAssetsToDocument", async (data) => {
        console.log("📥 Received addAssetsToDocument event");
        
        try {
            const { assets } = data;
            console.log(`Processing ${assets.length} assets...`);
            
            const insertionParent = addOnSandboxSdk.app.document;
            
            for (let i = 0; i < assets.length; i++) {
                const asset = assets[i];
                console.log(`Adding asset ${i + 1}/${assets.length}`);
                
                try {
                    const blob = await dataURItoBlob(asset.url);
                    await insertionParent.addImage(blob);
                    console.log(`✓ Asset ${i + 1} added`);
                } catch (error) {
                    console.error(`❌ Failed to add asset ${i + 1}:`, error);
                }
            }
            
            addOnSandboxSdk.instance.emit("assetsAdded", { 
                success: true,
                count: assets.length 
            });
            
            console.log("✓ All assets processed");
            
        } catch (error) {
            console.error("❌ Error in addAssetsToDocument:", error);
            addOnSandboxSdk.instance.emit("assetsAdded", { 
                success: false, 
                error: error.message 
            });
        }
    });
    
    console.log("✓ Document sandbox listeners ready");
});

async function dataURItoBlob(dataURI) {
    if (dataURI.startsWith('data:')) {
        const response = await fetch(dataURI);
        return await response.blob();
    }
    throw new Error("Invalid data URI");
}
