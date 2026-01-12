const pptxgen = require('pptxgenjs');
const html2pptx = require('/home/yuki/.claude/plugins/cache/anthropic-agent-skills/example-skills/69c0b1a06741/skills/pptx/scripts/html2pptx.js');
const path = require('path');

const SLIDES_DIR = '/home/yuki/lammps_settings_obata/hirataken20251122-2/workspace/slides';
const OUTPUT_DIR = '/home/yuki/lammps_settings_obata/hirataken20251122-2/outputs/fine_search_shoulder';

async function createPresentation() {
    const pptx = new pptxgen();
    pptx.layout = 'LAYOUT_16x9';
    pptx.title = '2元LJポテンシャルによる液体Gaの肩構造再現';
    pptx.author = 'Progress Report';

    // Slide 1: Title
    await html2pptx(path.join(SLIDES_DIR, 'slide1.html'), pptx);

    // Slide 2: Background
    await html2pptx(path.join(SLIDES_DIR, 'slide2.html'), pptx);

    // Slide 3: Method
    await html2pptx(path.join(SLIDES_DIR, 'slide3.html'), pptx);

    // Slide 4: Grid Search
    await html2pptx(path.join(SLIDES_DIR, 'slide4.html'), pptx);

    // Slide 5: Results - Best Fit with image
    const { slide: slide5, placeholders: ph5 } = await html2pptx(path.join(SLIDES_DIR, 'slide5.html'), pptx);
    if (ph5.length > 0) {
        slide5.addImage({
            path: path.join(OUTPUT_DIR, 'analysis/best_fit_overlay.png'),
            x: ph5[0].x,
            y: ph5[0].y,
            w: ph5[0].w,
            h: ph5[0].h
        });
    }

    // Slide 6: Results - Heatmap with image
    const { slide: slide6, placeholders: ph6 } = await html2pptx(path.join(SLIDES_DIR, 'slide6.html'), pptx);
    if (ph6.length > 0) {
        slide6.addImage({
            path: path.join(OUTPUT_DIR, 'analysis/rfactor_heatmap.png'),
            x: ph6[0].x,
            y: ph6[0].y,
            w: ph6[0].w,
            h: ph6[0].h
        });
    }

    // Slide 7: Voronoi with image
    const { slide: slide7, placeholders: ph7 } = await html2pptx(path.join(SLIDES_DIR, 'slide7.html'), pptx);
    if (ph7.length > 0) {
        slide7.addImage({
            path: path.join(OUTPUT_DIR, 'voronoi/voronoi_summary.png'),
            x: ph7[0].x,
            y: ph7[0].y,
            w: ph7[0].w,
            h: ph7[0].h
        });
    }

    // Slide 8: Conclusion
    await html2pptx(path.join(SLIDES_DIR, 'slide8.html'), pptx);

    // Save
    const outputPath = '/home/yuki/lammps_settings_obata/hirataken20251122-2/progress_report.pptx';
    await pptx.writeFile({ fileName: outputPath });
    console.log('Created: ' + outputPath);
}

createPresentation().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});
