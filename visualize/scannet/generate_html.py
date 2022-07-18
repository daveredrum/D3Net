import os, sys, argparse
import bs4
from omegaconf import OmegaConf
from glob import glob

sys.path.append(os.getcwd())
# sys.path.append("../..")


def get_cap_td(soup, cap):
    cap_box = soup.new_tag("td")
    cap_box.string = cap  
    return cap_box 


def get_video_td(soup, video_path, width="500px", height="500px"):
    video_div = soup.new_tag("td")   

    video_tag = soup.new_tag("video")
    video_tag["class"] = "lazy-video"
    video_tag["controls"] = "True"
    video_tag["autoplay"] = "True"
    video_tag["muted"] = "True"
    video_tag["loop"] = "True"
    video_tag["style"] = f"width:{width};height:{height}"
    
    source_link = soup.new_tag("source")
    source_link["data-src"] = video_path
    source_link["type"] = "video/mp4"
    video_tag.append(source_link)
    video_div.append(video_tag)
    
    return video_div


def generate_semantic_label_html(cfg):
    """
    Generate html to preview pose estimation
    """
    split = 'val'
    output_dir = cfg.WWW_PATH.scannet
    sem_dir = os.path.join(output_dir, 'sem_seg')
    table_temp_file = os.path.join(output_dir, 'table_template.html')
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    num_page = len(scene_ids) // 50
    # index_file = os.path.join(output_root, 'index.html')
    # index_doc = """<html><head><title> ... </title></head><body></body></html>"""
    # index_soup = bs4.BeautifulSoup(index_doc, features="html.parser")

    for p in range(num_page):
        p_scene_ids = scene_ids[p*50:(p+1)*50]
        with open(table_temp_file, 'r') as temp:
            txt = temp.read()
            soup = bs4.BeautifulSoup(txt, features="html.parser")
        
        for scene_id in p_scene_ids:
            rgb_video_path = os.path.relpath(glob(os.path.join(output_dir, f'rgb/{split}/{scene_id}/*.mp4'))[0], sem_dir)
            gt_sem_label_video_path = os.path.relpath(glob(os.path.join(sem_dir, f'gt_sem_label/{split}/{scene_id}/*.mp4'))[0], sem_dir)
            pred_sem_label_video_path = os.path.relpath(glob(os.path.join(sem_dir, f'pred_sem_label/{split}/{scene_id}/*.mp4'))[0], sem_dir)

            new_row = soup.new_tag("tr")
            soup.body.table.append(new_row)
            
            new_row.append(get_cap_td(soup, scene_id))
            new_row.append(get_video_td(soup, rgb_video_path))
            new_row.append(get_video_td(soup, gt_sem_label_video_path))
            new_row.append(get_video_td(soup, pred_sem_label_video_path))

        html_path = os.path.join(sem_dir, f"{p}.html")

        with open(html_path, "w") as f:
            f.write(str(soup))
            
    # with open(index_file, "w") as f:
    #     f.write(str(index_soup))
            
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', help='The data main folder')
    parser.add_argument('-o', '--output', help='The output folder', default="output")
    parser.add_argument('-i', '--instance_num_per_page', help='The number of mask instances per html page', default=100, type=int)
    parser.add_argument('-t', '--html_template_path', help='The html template')
    parser.add_argument('-j', '--json_path', help='The json file with shape sort result')
    
    parser.add_argument('-p', '--preview', default='', type=str, help='generate previews for semantic segmentation or ...: semantic | ...')
    args = parser.parse_args()
    cfg = OmegaConf.load('conf/path.yaml')
    
    if args.preview == 'semantic':
        generate_semantic_label_html(cfg)
    # elif args.preview == 'sort':
    #     generate_shape_sort_html(args)